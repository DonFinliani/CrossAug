import collections
import re
import string

from gfmrag.evaluation.base_evaluator import BaseEvaluator


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_tokens(s: str) -> list[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> tuple[float, float, float]:
    gold_toks = get_tokens(gold)
    pred_toks = get_tokens(pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        same = int(gold_toks == pred_toks)
        return same, same, same
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


class LiteraryQAEvaluator(BaseEvaluator):
    def evaluate(self) -> dict:
        metrics = {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        questions: list[str] = []
        predicted_answers: list[str] = []
        gold_answer_lists: list[list[str]] = []
        ids: list = []

        for pred in self.data:
            if "Answer: " in pred["response"]:
                pre_ans = pred["response"].split("Answer:")[1].strip()
            else:
                pre_ans = pred["response"]
            gold_answers = [pred["answer"]] + pred.get("answer_aliases", [])
            em = max(compute_exact(gold, pre_ans) for gold in gold_answers)
            f1, precision, recall = max(
                (compute_f1(gold, pre_ans) for gold in gold_answers),
                key=lambda item: item[0],
            )
            metrics["em"] += float(em)
            metrics["f1"] += f1
            metrics["precision"] += precision
            metrics["recall"] += recall
            questions.append(pred.get("question", ""))
            predicted_answers.append(pre_ans)
            gold_answer_lists.append(gold_answers)
            ids.append(pred.get("id", len(ids)))

        metrics = self.normalize_metrics_by_total_queries(metrics)
        metrics = self.maybe_evaluate_llm_answer_consistency(
            metrics=metrics,
            questions=questions,
            gold_answers=gold_answer_lists,
            predicted_answers=predicted_answers,
            ids=ids,
        )
        return metrics
