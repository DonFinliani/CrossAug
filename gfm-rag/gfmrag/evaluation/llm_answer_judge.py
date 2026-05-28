import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LLMAnswerJudge:
    """DeepSeek answer-consistency judge matching HippoRAG's implementation."""

    score_name = "LLMAnswerConsistency"

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        max_workers: int = 100,
        max_tokens: int = 5128,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
        seed: int = 42,
        cache_dir: str | None = None,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"LLM answer judge is enabled, but environment variable {api_key_env} is not set."
            )

        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.max_workers = max(1, int(max_workers or 1))
        self.max_tokens = int(max_tokens or 2048)
        self.max_retries = max(0, int(max_retries or 0))
        self.retry_sleep = max(0.0, float(retry_sleep or 0.0))
        self.seed = seed
        self.temperature = 0
        self.cache_dir = cache_dir or os.path.join("outputs", "llm_cache", "answer_judge")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(
            self.cache_dir,
            f"{self.model.replace('/', '_')}_answer_judge_cache.sqlite",
        )
        self._cache_lock = threading.Lock()
        self._init_cache()

        limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
        http_client = httpx.Client(
            limits=limits,
            timeout=httpx.Timeout(10 * 60, read=10 * 60),
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            http_client=http_client,
            max_retries=2,
        )

    def _init_cache(self) -> None:
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS answer_judge_cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def build_messages(
        question: str,
        gold_list: list[str],
        predicted: str,
    ) -> list[dict[str, str]]:
        gold_answers = list(gold_list or [])
        gold_answer_text = (
            gold_answers[0]
            if len(gold_answers) == 1
            else json.dumps(gold_answers, ensure_ascii=False)
        )
        return [
            {
                "role": "system",
                "content": "You are an expert evaluator.",
            },
            {
                "role": "user",
                "content": (
                    "Please evaluate if the generated answer is correct by comparing it with the gold answer.\n"
                    f"Question: {question or ''}\n"
                    f"Generated answer: {predicted or ''}\n"
                    f"Gold answer: {gold_answer_text}\n\n"
                    "The generated answer should be considered correct if it:\n"
                    "1. Contains the key information from the gold answer\n"
                    "2. Is factually accurate and consistent with the gold answer\n"
                    "3. Does not contain any contradicting information\n\n"
                    "If multiple gold answers are provided, the generated answer is correct if it satisfies "
                    "the criteria for any one of them.\n\n"
                    "Respond with ONLY 'correct' or 'incorrect'.\n"
                    "Response:"
                ),
            },
        ]

    @classmethod
    def parse_judge_response(cls, response: str) -> bool:
        if not response:
            logger.warning("Empty LLM answer judge response.")
            return False

        text = response.strip()
        normalized_label = re.sub(r"[^a-z]", "", text.lower())
        if normalized_label == "correct":
            return True
        if normalized_label == "incorrect":
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
                logger.warning("Could not parse LLM answer judge response: %s", response)
                return False
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning(
                    "Could not parse JSON object from LLM answer judge response: %s",
                    response,
                )
                return False

        if isinstance(payload, str):
            normalized_payload = re.sub(r"[^a-z]", "", payload.lower())
            if normalized_payload == "correct":
                return True
            if normalized_payload == "incorrect":
                return False
            logger.warning(
                "LLM answer judge string response has no recognized label: %s",
                response,
            )
            return False

        if not isinstance(payload, dict):
            logger.warning(
                "LLM answer judge response is not a recognized object or label: %s",
                response,
            )
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

        logger.warning(
            "LLM answer judge response has no recognized boolean key: %s",
            response,
        )
        return False

    @classmethod
    def is_parseable_response(cls, response: str) -> bool:
        if not response or not response.strip():
            return False

        text = response.strip()
        normalized_label = re.sub(r"[^a-z]", "", text.lower())
        if normalized_label in {"correct", "incorrect"}:
            return True

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return bool(
                    re.search(r"\bincorrect\b", text, flags=re.IGNORECASE)
                    or re.search(r"\bcorrect\b", text, flags=re.IGNORECASE)
                )
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return False

        if isinstance(payload, str):
            normalized_payload = re.sub(r"[^a-z]", "", payload.lower())
            return normalized_payload in {"correct", "incorrect"}

        if not isinstance(payload, dict):
            return False

        for key in ("same", "equivalent", "consistent", "is_same", "is_equivalent"):
            if key in payload:
                return True

        for key in ("answer", "judgment", "judgement", "verdict", "label", "result"):
            value = str(payload.get(key, "")).strip().lower()
            if value in {"correct", "incorrect"}:
                return True

        return False

    def _cache_key(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "temperature": self.temperature,
            "messages": messages,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_cached(self, cache_key: str) -> tuple[str, dict[str, Any]] | None:
        with self._cache_lock:
            with sqlite3.connect(self.cache_path) as conn:
                row = conn.execute(
                    "SELECT response, metadata FROM answer_judge_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
        if row is None:
            return None
        return row[0], json.loads(row[1])

    def _set_cached(
        self,
        cache_key: str,
        response: str,
        metadata: dict[str, Any],
    ) -> None:
        with self._cache_lock:
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO answer_judge_cache
                    (cache_key, response, metadata, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cache_key, response, json.dumps(metadata), time.time()),
                )

    def _retry_delay(self, attempt_idx: int) -> float:
        if self.retry_sleep <= 0:
            return 0.0
        delay = self.retry_sleep * (2 ** max(0, attempt_idx - 1))
        jitter = random.uniform(0, min(0.5, self.retry_sleep / 2))
        return delay + jitter

    def _call_llm_once(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "temperature": self.temperature,
            "reasoning_effort": "high",
            "extra_body": {"thinking": {"type": "enabled"}}
        }
        response = self.client.chat.completions.create(**params)
        message = response.choices[0].message
        response_message = message.content or ""
        reasoning_content = getattr(message, "reasoning_content", None)
        usage = response.usage
        metadata = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "finish_reason": response.choices[0].finish_reason,
            "content_empty": not bool(response_message.strip()),
            "reasoning_content_present": bool(reasoning_content),
            "reasoning_content_chars": len(reasoning_content or ""),
        }
        return response_message, metadata

    def _call_llm(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        last_response = ""
        last_metadata: dict[str, Any] = {}
        errors: list[str] = []
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                raw_response, metadata = self._call_llm_once(messages)
                metadata = dict(metadata or {})
                metadata["attempt"] = attempt
                metadata["max_attempts"] = total_attempts
                last_response = raw_response
                last_metadata = metadata
                if self.is_parseable_response(raw_response):
                    metadata["judge_failed"] = False
                    metadata["retry_errors"] = errors
                    return raw_response, metadata

                reason = "empty_response" if not raw_response.strip() else "unparseable_response"
                errors.append(reason)
                logger.warning(
                    "LLM answer judge returned %s on attempt %s/%s.",
                    reason,
                    attempt,
                    total_attempts,
                )
            except Exception as exc:  # noqa: BLE001 - preserve failed judge diagnostics
                errors.append(repr(exc))
                last_metadata = {
                    "attempt": attempt,
                    "max_attempts": total_attempts,
                    "exception": repr(exc),
                }
                logger.warning(
                    "LLM answer judge request failed on attempt %s/%s: %r",
                    attempt,
                    total_attempts,
                    exc,
                )

            if attempt < total_attempts:
                delay = self._retry_delay(attempt)
                if delay:
                    time.sleep(delay)

        last_metadata = dict(last_metadata or {})
        last_metadata["attempt"] = total_attempts
        last_metadata["max_attempts"] = total_attempts
        last_metadata["judge_failed"] = True
        last_metadata["retry_errors"] = errors
        return last_response, last_metadata

    def score_one(
        self,
        question: str,
        gold_list: list[str],
        predicted: str,
    ) -> dict[str, Any]:
        messages = self.build_messages(
            question=question,
            gold_list=gold_list,
            predicted=predicted,
        )
        cache_key = self._cache_key(messages)
        cached = self._get_cached(cache_key)
        if cached is None:
            raw_response, metadata = self._call_llm(messages)
            metadata = dict(metadata or {})
            metadata["cache_hit"] = False
        else:
            raw_response, metadata = cached
            metadata = dict(metadata or {})
            if self.is_parseable_response(raw_response):
                metadata["cache_hit"] = True
            else:
                raw_response, metadata = self._call_llm(messages)
                metadata = dict(metadata or {})
                metadata["cache_hit"] = False
                metadata["stale_unparseable_cache_ignored"] = True

        if not metadata.get("judge_failed"):
            self._set_cached(cache_key, raw_response, metadata)

        judge_failed = bool(metadata.get("judge_failed")) or not self.is_parseable_response(
            raw_response
        )
        same = False if judge_failed else self.parse_judge_response(raw_response)
        return {
            self.score_name: 1.0 if same else 0.0,
            "judge_model": self.model,
            "judge_response": raw_response,
            "judge_metadata": metadata,
            "judge_failed": judge_failed,
        }

    def score_batch(
        self,
        questions: list[str],
        gold_answers: list[list[str]],
        predicted_answers: list[str],
        total_query_count: int | None = None,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        assert len(gold_answers) == len(predicted_answers), (
            "Length of gold answers and predicted answers should be the same."
        )
        assert len(questions) == len(predicted_answers), (
            "Length of questions and predicted answers should be the same."
        )

        denominator = int(total_query_count or len(predicted_answers))
        if denominator < len(predicted_answers):
            raise ValueError(
                "total_query_count cannot be smaller than the number of "
                f"predicted answers: total_query_count={denominator}, "
                f"predicted_answers={len(predicted_answers)}."
            )

        if not predicted_answers:
            return {self.score_name: 0.0}, []

        example_results: list[dict[str, Any] | None] = [None] * len(predicted_answers)
        max_workers = min(self.max_workers, len(predicted_answers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.score_one,
                    questions[idx],
                    gold_answers[idx],
                    predicted_answers[idx],
                ): idx
                for idx in range(len(predicted_answers))
            }
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="LLM Answer Judge",
            ):
                idx = future_to_idx[future]
                example_results[idx] = future.result()

        results = [result for result in example_results if result is not None]
        valid_results = [result for result in results if not result.get("judge_failed")]
        avg_score = sum(result[self.score_name] for result in results) / denominator
        metrics = {
            self.score_name: avg_score,
            "LLMAnswerJudgeFailureRate": 1.0 - (len(valid_results) / len(results)),
            "LLMAnswerJudgeCount": len(results),
        }
        if valid_results:
            metrics[f"{self.score_name}_valid_only"] = sum(
                result[self.score_name] for result in valid_results
            ) / len(valid_results)
        else:
            metrics[f"{self.score_name}_valid_only"] = 0.0
        return metrics, results
