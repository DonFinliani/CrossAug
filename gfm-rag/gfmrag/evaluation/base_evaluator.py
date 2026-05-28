import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from gfmrag.evaluation.llm_answer_judge import LLMAnswerJudge

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Base evaluator class for evaluation tasks.

    This abstract base class provides a foundation for implementing evaluators
    that assess model predictions. It handles loading prediction data from a JSON
    lines file where each line contains a single JSON object.

    Args:
        prediction_file (str): Path to the JSON lines prediction file to evaluate.
            Each line should contain a valid JSON object.

    Attributes:
        data (List[dict]): List of prediction data loaded from the JSON lines file.

    Examples:
        >>> evaluator = MyEvaluator("predictions.jsonl")
        >>> results = evaluator.evaluate()

    Note:
        Subclasses must implement the `evaluate()` method to define evaluation logic.
    """

    def __init__(
        self,
        prediction_file: str,
        output_dir: str | None = None,
        seed: int = 42,
        enable_llm_answer_judge: bool = False,
        answer_judge_model: str = "deepseek-v4-pro",
        answer_judge_base_url: str = "https://api.deepseek.com",
        answer_judge_api_key_env: str = "DEEPSEEK_API_KEY",
        answer_judge_max_workers: int = 100,
        answer_judge_max_tokens: int = 5128,
        answer_judge_max_retries: int = 3,
        answer_judge_retry_sleep: float = 2.0,
        answer_judge_cache_dir: str | None = None,
        total_query_count: int | None = None,
    ) -> None:
        super().__init__()
        self.prediction_file = prediction_file
        self.output_dir = output_dir or os.path.dirname(os.path.abspath(prediction_file))
        self.seed = seed
        self.enable_llm_answer_judge = enable_llm_answer_judge
        self.answer_judge_model = answer_judge_model
        self.answer_judge_base_url = answer_judge_base_url
        self.answer_judge_api_key_env = answer_judge_api_key_env
        self.answer_judge_max_workers = answer_judge_max_workers
        self.answer_judge_max_tokens = answer_judge_max_tokens
        self.answer_judge_max_retries = answer_judge_max_retries
        self.answer_judge_retry_sleep = answer_judge_retry_sleep
        self.answer_judge_cache_dir = answer_judge_cache_dir
        with open(prediction_file) as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.raw_prediction_count = len(self.data)
        self.data = self._deduplicate_predictions(self.data)
        self.num_duplicate_predictions = self.raw_prediction_count - len(self.data)
        if not self.data:
            raise ValueError(
                f"No predictions found in {prediction_file}. "
                "The QA inference step likely produced zero test samples or all "
                "answer generation calls failed."
            )
        if total_query_count is None:
            self.total_query_count = len(self.data)
        else:
            self.total_query_count = int(total_query_count)
            if self.total_query_count < len(self.data):
                raise ValueError(
                    "total_query_count cannot be smaller than the number of "
                    f"predictions: total_query_count={self.total_query_count}, "
                    f"predictions={len(self.data)}."
                )
            if self.total_query_count <= 0:
                raise ValueError(
                    f"total_query_count must be positive, got {self.total_query_count}."
                )

    @staticmethod
    def _deduplicate_predictions(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen_ids = set()
        deduped_data = []
        duplicate_count = 0
        for row in data:
            if "id" not in row:
                deduped_data.append(row)
                continue

            row_id = row["id"]
            if row_id in seen_ids:
                duplicate_count += 1
                continue

            seen_ids.add(row_id)
            deduped_data.append(row)

        if duplicate_count:
            logger.warning(
                "Ignored %s duplicate prediction rows while evaluating %s rows.",
                duplicate_count,
                len(data),
            )
        return deduped_data

    def normalize_metrics_by_total_queries(self, metrics: dict[str, float]) -> dict:
        for k in list(metrics.keys()):
            metrics[k] /= self.total_query_count
        metrics["num_total_queries"] = self.total_query_count
        metrics["num_predictions"] = len(self.data)
        metrics["num_raw_predictions"] = self.raw_prediction_count
        metrics["num_duplicate_predictions"] = self.num_duplicate_predictions
        metrics["num_missing_predictions"] = self.total_query_count - len(self.data)
        return metrics

    def maybe_evaluate_llm_answer_consistency(
        self,
        metrics: dict[str, float],
        questions: list[str],
        gold_answers: list[list[str]],
        predicted_answers: list[str],
        ids: list[Any] | None = None,
    ) -> dict[str, float]:
        if not self.enable_llm_answer_judge:
            return metrics

        cache_dir = self.answer_judge_cache_dir or os.path.join(
            self.output_dir,
            "llm_cache",
            "answer_judge",
        )
        judge = LLMAnswerJudge(
            model=self.answer_judge_model,
            base_url=self.answer_judge_base_url,
            api_key_env=self.answer_judge_api_key_env,
            max_workers=self.answer_judge_max_workers,
            max_tokens=self.answer_judge_max_tokens,
            max_retries=self.answer_judge_max_retries,
            retry_sleep=self.answer_judge_retry_sleep,
            seed=self.seed,
            cache_dir=cache_dir,
        )
        judge_metrics, example_results = judge.score_batch(
            questions=questions,
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
            total_query_count=self.total_query_count,
        )
        metrics.update(judge_metrics)
        self._write_llm_answer_judge_details(
            questions=questions,
            gold_answers=gold_answers,
            predicted_answers=predicted_answers,
            example_results=example_results,
            ids=ids,
        )
        return metrics

    def _write_llm_answer_judge_details(
        self,
        questions: list[str],
        gold_answers: list[list[str]],
        predicted_answers: list[str],
        example_results: list[dict[str, Any]],
        ids: list[Any] | None = None,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "llm_answer_judge.jsonl")
        ids = ids or list(range(len(predicted_answers)))
        with open(output_path, "w") as f:
            for idx, result in enumerate(example_results):
                row = {
                    "id": ids[idx],
                    "question": questions[idx],
                    "gold_answers": gold_answers[idx],
                    "predicted_answer": predicted_answers[idx],
                }
                row.update(result)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @abstractmethod
    def evaluate(self) -> dict:
        pass
