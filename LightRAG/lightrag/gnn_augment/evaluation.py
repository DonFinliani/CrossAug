from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import string
from collections import Counter
from typing import Any

import numpy as np

from lightrag.utils import is_timeout_exception, logger

from .literaryqa import gold_answers_for_sample


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def exact_match_score(gold_answers: list[str], prediction: str) -> float:
    if not gold_answers:
        return 0.0
    return float(
        max(
            normalize_answer(gold) == normalize_answer(prediction)
            for gold in gold_answers
        )
    )


def f1_score(gold_answers: list[str], prediction: str) -> float:
    def _one(gold: str) -> float:
        gold_tokens = normalize_answer(gold).split()
        pred_tokens = normalize_answer(prediction).split()
        if not gold_tokens or not pred_tokens:
            return float(gold_tokens == pred_tokens)
        common = Counter(gold_tokens) & Counter(pred_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    if not gold_answers:
        return 0.0
    return float(max(_one(gold) for gold in gold_answers))


def parse_judge_response(response: str) -> bool:
    if not response:
        return False
    text = response.strip()
    normalized = re.sub(r"[^a-z]", "", text.lower())
    if normalized == "correct":
        return True
    if normalized == "incorrect":
        return False

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            if re.search(r"\bincorrect\b", text, flags=re.IGNORECASE):
                return False
            if re.search(r"\bcorrect\b", text, flags=re.IGNORECASE):
                return True
            return False
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return False

    if isinstance(payload, str):
        return parse_judge_response(payload)
    if not isinstance(payload, dict):
        return False
    for key in ("same", "equivalent", "consistent", "is_same", "is_equivalent"):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) > 0
        if isinstance(value, str):
            return value.strip().lower() in {
                "true",
                "yes",
                "y",
                "1",
                "same",
                "equivalent",
                "consistent",
            }
    for key in ("answer", "judgment", "judgement", "verdict", "label", "result"):
        if key not in payload:
            continue
        value = str(payload[key]).strip().lower()
        if value == "correct":
            return True
        if value == "incorrect":
            return False
    return False


class DeepSeekAnswerJudge:
    def __init__(
        self,
        *,
        model: str = "deepseek-v4-pro",
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        max_tokens: int = 512,
        concurrency: int = 4,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_path: str | None = None,
        enable_cache: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.concurrency = max(1, concurrency)
        self.max_retries = max(0, max_retries)
        self.retry_delay = max(0.0, retry_delay)
        self.cache_path = cache_path or os.getenv("ANSWER_JUDGE_CACHE_PATH")
        self.enable_cache = enable_cache and bool(self.cache_path)
        self._cache_lock = asyncio.Lock()
        self._cache: dict[str, Any] = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        if not self.enable_cache or not self.cache_path or not os.path.exists(self.cache_path):
            return {"version": 1, "entries": {}}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError("judge cache root is not a JSON object")
            entries = payload.get("entries", {})
            if not isinstance(entries, dict):
                raise ValueError("judge cache entries is not a JSON object")
            payload["version"] = int(payload.get("version", 1))
            payload["entries"] = entries
            logger.info("Loaded DeepSeek judge cache: %s entries from %s", len(entries), self.cache_path)
            return payload
        except Exception as exc:
            logger.warning("Ignoring unreadable DeepSeek judge cache %s: %s", self.cache_path, exc)
            return {"version": 1, "entries": {}}

    def _save_cache_unlocked(self) -> None:
        if not self.enable_cache or not self.cache_path:
            return
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        tmp_path = f"{self.cache_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.cache_path)

    def _cache_key(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "cache_version": "lightrag-deepseek-answer-judge-v1",
            "model": self.model,
            "base_url": self.base_url,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        cache_text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(cache_text.encode("utf-8")).hexdigest()

    async def _get_cached_result(self, cache_key: str) -> dict[str, Any] | None:
        if not self.enable_cache:
            return None
        async with self._cache_lock:
            entry = self._cache.get("entries", {}).get(cache_key)
            if not isinstance(entry, dict):
                return None
            result = entry.get("result")
            if not isinstance(result, dict):
                return None
            cached = dict(result)
            cached["judge_cache_hit"] = True
            cached["judge_cache_key"] = cache_key
            cached["judge_attempts"] = 0
            return cached

    async def _store_cached_result(self, cache_key: str, result: dict[str, Any]) -> None:
        if not self.enable_cache:
            return
        if result.get("judge_skipped") or result.get("judge_error"):
            return
        entry_result = dict(result)
        entry_result["judge_cache_hit"] = False
        entry_result["judge_cache_key"] = cache_key
        async with self._cache_lock:
            self._cache.setdefault("entries", {})[cache_key] = {
                "result": entry_result,
                "model": self.model,
                "base_url": self.base_url,
                "max_tokens": self.max_tokens,
            }
            self._save_cache_unlocked()

    @staticmethod
    def _field(value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)

    @classmethod
    def _response_text(cls, response: Any) -> str:
        choices = cls._field(response, "choices")
        if not choices:
            raise ValueError("judge response missing choices")
        choice = choices[0]
        message = cls._field(choice, "message")
        content = cls._field(message, "content") if message is not None else None
        if content is None:
            content = cls._field(choice, "text")
        if content is None:
            raise ValueError("judge response missing message content")
        return str(content)

    def _failure_result(self, exc: Exception) -> dict[str, Any]:
        return {
            "LLMAnswerConsistency": 0.0,
            "judge_model": self.model,
            "judge_response": "",
            "judge_error_type": type(exc).__name__,
            "judge_error": str(exc),
            "judge_skipped": True,
        }

    @classmethod
    def _status_code(cls, exc: Exception) -> int | None:
        for value in (
            cls._field(exc, "status_code"),
            cls._field(exc, "code"),
            cls._field(cls._field(exc, "response"), "status_code"),
        ):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _is_retryable_exception(cls, exc: Exception) -> bool:
        if is_timeout_exception(exc):
            return True
        status_code = cls._status_code(exc)
        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "service_unavailable",
                "service is too busy",
                "temporarily",
                "rate limit",
                "timeout",
            )
        )

    def _messages(
        self,
        question: str,
        gold_answers: list[str],
        prediction: str,
    ) -> list[dict[str, str]]:
        gold_text = (
            gold_answers[0]
            if len(gold_answers) == 1
            else json.dumps(gold_answers, ensure_ascii=False)
        )
        return [
            {"role": "system", "content": "You are an expert evaluator."},
            {
                "role": "user",
                "content": (
                    "Please evaluate if the generated answer is correct by comparing it "
                    "with the gold answer.\n"
                    f"Question: {question or ''}\n"
                    f"Generated answer: {prediction or ''}\n"
                    f"Gold answer: {gold_text}\n\n"
                    "The generated answer should be considered correct if it contains "
                    "the key information from the gold answer, is factually accurate, "
                    "and does not contradict the gold answer. If multiple gold answers "
                    "are provided, satisfying any one of them is enough.\n\n"
                    "Respond with ONLY 'correct' or 'incorrect'."
                ),
            },
        ]

    async def score_many(
        self,
        questions: list[str],
        gold_answers: list[list[str]],
        predictions: list[str],
    ) -> list[dict[str, Any]]:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "DeepSeek LLM-as-judge requires the openai package. Install with "
                "`pip install openai` or `pip install -e .[gnn]`."
            ) from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"DeepSeek judge enabled but environment variable {self.api_key_env} is not set."
            )

        client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _score(idx: int) -> dict[str, Any]:
            messages = self._messages(questions[idx], gold_answers[idx], predictions[idx])
            cache_key = self._cache_key(messages)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result

            async with semaphore:
                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=0,
                            max_tokens=self.max_tokens,
                            stream=False,    
                            reasoning_effort="high",
                            extra_body={"thinking": {"type": "enabled"}},
                        )
                        raw = self._response_text(response)
                        result = {
                            "LLMAnswerConsistency": 1.0
                            if parse_judge_response(raw)
                            else 0.0,
                            "judge_model": self.model,
                            "judge_response": raw,
                            "judge_attempts": attempt + 1,
                            "judge_cache_hit": False,
                            "judge_cache_key": cache_key,
                        }
                        await self._store_cached_result(cache_key, result)
                        return result
                    except Exception as exc:
                        if attempt < self.max_retries and self._is_retryable_exception(exc):
                            sleep_time = self.retry_delay * (2**attempt)
                            logger.warning(
                                "DeepSeek judge call %s failed on attempt %s/%s; "
                                "retrying in %.2fs: %s",
                                idx,
                                attempt + 1,
                                self.max_retries + 1,
                                sleep_time,
                                exc,
                            )
                            if sleep_time > 0:
                                await asyncio.sleep(sleep_time)
                            continue

                        if is_timeout_exception(exc):
                            logger.warning(
                                "Skipping timed-out DeepSeek judge call %s after %s attempts: %s",
                                idx,
                                attempt + 1,
                                exc,
                            )
                        else:
                            logger.warning(
                                "Skipping failed DeepSeek judge call %s after %s attempts: %s",
                                idx,
                                attempt + 1,
                                exc,
                            )
                        result = self._failure_result(exc)
                        result["judge_attempts"] = attempt + 1
                        result["judge_status_code"] = self._status_code(exc)
                        result["judge_cache_hit"] = False
                        result["judge_cache_key"] = cache_key
                        return result

                fallback = RuntimeError("DeepSeek judge failed without captured exception")
                result = self._failure_result(fallback)
                result["judge_cache_hit"] = False
                result["judge_cache_key"] = cache_key
                return result

        try:
            return await asyncio.gather(*[_score(idx) for idx in range(len(predictions))])
        finally:
            await client.close()


async def evaluate_literaryqa_predictions(
    samples: list[dict[str, Any]],
    predictions: list[str],
    *,
    enable_judge: bool = False,
    judge_model: str = "deepseek-v4-pro",
    judge_base_url: str = "https://api.deepseek.com",
    judge_api_key_env: str = "DEEPSEEK_API_KEY",
    judge_max_tokens: int = 512,
    judge_concurrency: int = 4,
    judge_max_retries: int = 3,
    judge_retry_delay: float = 1.0,
    judge_cache_path: str | None = None,
    enable_judge_cache: bool = True,
) -> dict[str, Any]:
    if len(samples) != len(predictions):
        raise ValueError("samples and predictions must have the same length.")

    golds = [gold_answers_for_sample(sample) for sample in samples]
    questions = [str(sample.get("question", "")) for sample in samples]
    per_example: list[dict[str, Any]] = []
    for sample, gold, prediction in zip(samples, golds, predictions):
        per_example.append(
            {
                "id": sample.get("id"),
                "question": sample.get("question"),
                "gold_answers": gold,
                "prediction": prediction,
                "ExactMatch": exact_match_score(gold, prediction),
                "F1": f1_score(gold, prediction),
            }
        )

    if enable_judge and samples:
        judge = DeepSeekAnswerJudge(
            model=judge_model,
            base_url=judge_base_url,
            api_key_env=judge_api_key_env,
            max_tokens=judge_max_tokens,
            concurrency=judge_concurrency,
            max_retries=judge_max_retries,
            retry_delay=judge_retry_delay,
            cache_path=judge_cache_path,
            enable_cache=enable_judge_cache,
        )
        judge_scores = await judge.score_many(questions, golds, predictions)
        for row, judge_row in zip(per_example, judge_scores):
            row.update(judge_row)
    elif enable_judge:
        logger.info("DeepSeek judge enabled, but no samples were provided.")

    aggregate = {
        "ExactMatch": float(np.mean([row["ExactMatch"] for row in per_example])) if per_example else 0.0,
        "F1": float(np.mean([row["F1"] for row in per_example])) if per_example else 0.0,
    }
    if enable_judge:
        aggregate["LLMAnswerConsistency"] = (
            float(np.mean([row.get("LLMAnswerConsistency", 0.0) for row in per_example]))
            if per_example
            else 0.0
        )
        aggregate["judge_error_count"] = sum(
            1
            for row in per_example
            if row.get("judge_skipped") or row.get("judge_error")
        )
        aggregate["judge_cache_hits"] = sum(
            1 for row in per_example if row.get("judge_cache_hit")
        )
        if judge_cache_path:
            aggregate["judge_cache_path"] = judge_cache_path

    return {
        "aggregate": aggregate,
        "examples": per_example,
    }
