import copy
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Dict, Tuple, Optional, Union, Callable
from collections import Counter
import numpy as np
from tqdm import tqdm

from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
from ..utils.eval_utils import normalize_answer

logger = get_logger(__name__)

# Reference: MRQA official eval
class QAExactMatch(BaseMetric):
    metric_name: str = "qa_exact_match"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the Exact Match (EM) score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged EM score.
                - A list of dictionaries with EM scores for each example.
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_em = 0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
            aggregated_em = aggregation_fn(em_scores)
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}

        return pooled_eval_results, example_eval_results

class QAF1Score(BaseMetric):
    metric_name: str = "qa_f1_score"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the F1 score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged F1 score.
                - A list of dictionaries with F1 scores for each example.
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            return 2 * (precision * recall) / (precision + recall)

        example_eval_results = []
        total_f1 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
            aggregated_f1 = aggregation_fn(f1_scores)
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}

        return pooled_eval_results, example_eval_results


class QALLMAnswerJudge(BaseMetric):
    metric_name: str = "qa_llm_answer_consistency"
    score_name: str = "LLMAnswerConsistency"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)
        self.max_workers = max(1, int(getattr(self.global_config, "answer_judge_max_workers", 8) or 1))
        self.llm_model = self._build_llm_model()

    def _build_llm_model(self) -> Any:
        from ..llm.openai_gpt import CacheOpenAI

        api_key_env = getattr(self.global_config, "answer_judge_api_key_env", "DEEPSEEK_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"LLM answer judge is enabled, but environment variable {api_key_env} is not set."
            )

        judge_config = copy.copy(self.global_config)
        judge_config.llm_name = getattr(self.global_config, "answer_judge_model", "deepseek-v4-pro")
        judge_config.llm_base_url = getattr(self.global_config, "answer_judge_base_url", "https://api.deepseek.com")
        judge_config.max_new_tokens = int(getattr(self.global_config, "answer_judge_max_tokens", 512) or 512)
        judge_config.num_gen_choices = 1
        judge_config.seed = getattr(self.global_config, "seed", 42)
        judge_config.temperature = 0
        judge_config.azure_endpoint = None
        judge_config.llm_api_key = api_key

        cache_dir = getattr(self.global_config, "answer_judge_cache_dir", None) or os.path.join(
            judge_config.save_dir or "outputs",
            "llm_cache",
            "answer_judge",
        )
        cache_filename = f"{judge_config.llm_name.replace('/', '_')}_answer_judge_cache.sqlite"
        return CacheOpenAI(
            cache_dir=cache_dir,
            global_config=judge_config,
            cache_filename=cache_filename,
            high_throughput=True,
        )

    @staticmethod
    def _build_messages(question: str, gold_list: List[str], predicted: str) -> List[Dict[str, str]]:
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
                logger.warning("Could not parse JSON object from LLM answer judge response: %s", response)
                return False

        if isinstance(payload, str):
            normalized_payload = re.sub(r"[^a-z]", "", payload.lower())
            if normalized_payload == "correct":
                return True
            if normalized_payload == "incorrect":
                return False
            logger.warning("LLM answer judge string response has no recognized label: %s", response)
            return False

        if not isinstance(payload, dict):
            logger.warning("LLM answer judge response is not a recognized object or label: %s", response)
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
                return value.strip().lower() in {"true", "yes", "y", "1", "same", "equivalent", "consistent"}

        for key in ("answer", "judgment", "judgement", "verdict", "label", "result"):
            if key not in payload:
                continue
            value = str(payload[key]).strip().lower()
            if value == "correct":
                return True
            if value == "incorrect":
                return False

        logger.warning("LLM answer judge response has no recognized boolean key: %s", response)
        return False

    def _score_one(self, question: str, gold_list: List[str], predicted: str) -> Dict[str, Union[float, str, Dict]]:
        messages = self._build_messages(question=question, gold_list=gold_list, predicted=predicted)
        try:
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
                temperature=0,
                seed=getattr(self.global_config, "seed", 42),
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
                n=None,
            )
        except Exception as exc:
            logger.warning("LLM answer judge request failed after retries; marking this example as incorrect: %s", exc)
            return {
                self.score_name: 0.0,
                "judge_model": self.llm_model.llm_name,
                "judge_response": "",
                "judge_metadata": {
                    "cache_hit": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            }
        metadata = dict(metadata or {})
        metadata["cache_hit"] = cache_hit
        same = self.parse_judge_response(raw_response)
        return {
            self.score_name: 1.0 if same else 0.0,
            "judge_model": self.llm_model.llm_name,
            "judge_response": raw_response,
            "judge_metadata": metadata,
        }

    def calculate_metric_scores(
        self,
        gold_answers: List[List[str]],
        predicted_answers: List[str],
        questions: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, float], List[Dict[str, Union[float, str, Dict]]]]:
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."
        if questions is None:
            questions = [""] * len(predicted_answers)
        assert len(questions) == len(predicted_answers), "Length of questions and predicted answers should be the same."

        if not predicted_answers:
            return {self.score_name: 0.0}, []

        example_eval_results = [None] * len(predicted_answers)
        max_workers = min(self.max_workers, len(predicted_answers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._score_one, questions[idx], gold_answers[idx], predicted_answers[idx]): idx
                for idx in range(len(predicted_answers))
            }
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="LLM Answer Judge",
            ):
                idx = future_to_idx[future]
                try:
                    example_eval_results[idx] = future.result()
                except Exception as exc:
                    logger.warning(
                        "Unexpected LLM answer judge worker failure; marking this example as incorrect: %s",
                        exc,
                    )
                    example_eval_results[idx] = {
                        self.score_name: 0.0,
                        "judge_model": getattr(self.llm_model, "llm_name", None),
                        "judge_response": "",
                        "judge_metadata": {
                            "cache_hit": False,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    }

        avg_score = sum(result[self.score_name] for result in example_eval_results) / len(example_eval_results)
        return {self.score_name: avg_score}, example_eval_results
