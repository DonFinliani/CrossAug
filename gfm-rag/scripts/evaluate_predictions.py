#!/usr/bin/env python3
"""Evaluate an existing GFM-RAG prediction.jsonl without rerunning inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gfmrag.evaluation import (
    HotpotQAEvaluator,
    LiteraryQAEvaluator,
    MusiqueEvaluator,
    TwoWikiQAEvaluator,
)


EVALUATORS = {
    "hotpotqa": HotpotQAEvaluator,
    "musique": MusiqueEvaluator,
    "2wikimultihopqa": TwoWikiQAEvaluator,
    "two_wiki": TwoWikiQAEvaluator,
    "literaryqa": LiteraryQAEvaluator,
}


def count_test_file(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return sum(1 for line in f if line.strip())
        data = json.load(f)

    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("data", "samples", "examples", "questions"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
    raise ValueError(f"Could not count queries from unsupported test file: {path}")


def resolve_maybe_relative_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def stage1_test_path(dataset_root: Path, data_name: str) -> Path:
    return dataset_root / data_name / "processed" / "stage1" / "test.json"


def read_yaml_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def hydra_runtime_cwd(run_dir: Path) -> Path:
    hydra_yaml = run_dir / ".hydra" / "hydra.yaml"
    if not hydra_yaml.exists():
        return REPO_ROOT
    try:
        hydra_cfg = read_yaml_file(hydra_yaml)
    except Exception:
        return REPO_ROOT
    cwd = hydra_cfg.get("hydra", {}).get("runtime", {}).get("cwd")
    return Path(cwd).resolve() if cwd else REPO_ROOT


def infer_data_name_from_prediction_path(prediction_file: Path) -> str | None:
    parts = prediction_file.parts
    for idx, part in enumerate(parts):
        if part == "qa_inference" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def candidate_test_files(args: argparse.Namespace, prediction_file: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []

    if args.test_file:
        candidates.append(
            (
                "--test-file",
                resolve_maybe_relative_path(args.test_file, Path.cwd()),
            )
        )

    run_dir = prediction_file.parent
    hydra_config = run_dir / ".hydra" / "config.yaml"
    if hydra_config.exists():
        try:
            cfg = read_yaml_file(hydra_config)
            dataset_cfg = cfg.get("dataset", {})
            data_name = dataset_cfg.get("data_name")
            dataset_root = dataset_cfg.get("root")
            if data_name and dataset_root:
                runtime_cwd = hydra_runtime_cwd(run_dir)
                root = resolve_maybe_relative_path(dataset_root, runtime_cwd)
                candidates.append(
                    (
                        f"Hydra config dataset.root/data_name ({hydra_config})",
                        stage1_test_path(root, data_name),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not inspect Hydra config {hydra_config}: {exc}")

    explicit_roots: list[Path] = []
    if args.dataset_root:
        explicit_roots.append(resolve_maybe_relative_path(args.dataset_root, Path.cwd()))
    else:
        explicit_roots.extend(
            [
                REPO_ROOT / "data" / "hipporag_stage1_exports",
                REPO_ROOT / "data",
            ]
        )

    data_names: list[str] = []
    if args.data_name:
        data_names.append(args.data_name)
    inferred_data_name = infer_data_name_from_prediction_path(prediction_file)
    if inferred_data_name:
        data_names.append(inferred_data_name)
    dataset_default = "2wikimultihopqa" if args.dataset == "two_wiki" else args.dataset
    data_names.extend([f"{dataset_default}_test", dataset_default])

    seen_pairs: set[tuple[str, str]] = set()
    for root in explicit_roots:
        for data_name in data_names:
            pair = (str(root), data_name)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            candidates.append(
                (
                    f"dataset root/data_name ({root}, {data_name})",
                    stage1_test_path(root, data_name),
                )
            )

    return candidates


def infer_total_query_count(
    args: argparse.Namespace,
    prediction_file: Path,
) -> tuple[int | None, str]:
    if args.total_query_count is not None:
        return int(args.total_query_count), "--total-query-count"

    if args.no_auto_total_query_count:
        return None, "disabled; using prediction row count"

    tried: list[str] = []
    seen_paths: set[Path] = set()
    for source, path in candidate_test_files(args, prediction_file):
        path = path.resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        tried.append(f"{source}: {path}")
        if path.exists():
            return count_test_file(path), f"{source}: {path}"

    if tried:
        print("[WARN] Could not infer total query count from candidate test files:")
        for item in tried:
            print(f"[WARN]   {item}")
    return None, "prediction row count fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the QA evaluation step for an existing prediction.jsonl."
    )
    parser.add_argument(
        "--prediction-file",
        required=True,
        help="Path to prediction.jsonl produced by stage3_qa_inference.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(EVALUATORS),
        help="Evaluator to use.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for metrics.json and llm_answer_judge.jsonl. "
            "Defaults to the prediction file directory."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-llm-answer-judge",
        action="store_true",
        default=True,
        help="Only compute exact-match/F1-style metrics.",
    )
    parser.add_argument(
        "--enable-llm-answer-judge",
        dest="disable_llm_answer_judge",
        action="store_false",
        help="Also compute LLM-based answer consistency metrics.",
    )
    parser.add_argument(
        "--total-query-count",
        type=int,
        default=None,
        help=(
            "Denominator for final metrics. Missing predictions are counted as "
            "zero. If omitted, the script tries to infer it from stage1 test.json."
        ),
    )
    parser.add_argument(
        "--no-auto-total-query-count",
        action="store_true",
        help="Disable automatic denominator inference and use prediction row count.",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help="Optional explicit test.json/jsonl path used to infer total query count.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help=(
            "Optional dataset root containing <data-name>/processed/stage1/test.json. "
            "Defaults to data/hipporag_stage1_exports and data under the repo."
        ),
    )
    parser.add_argument(
        "--data-name",
        default=None,
        help="Optional stage1 data directory name, e.g. 2wikimultihopqa_base.",
    )
    parser.add_argument("--answer-judge-model", default="deepseek-v4-pro")
    parser.add_argument("--answer-judge-base-url", default="https://api.deepseek.com")
    parser.add_argument("--answer-judge-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--answer-judge-max-workers", type=int, default=100)
    parser.add_argument("--answer-judge-max-tokens", type=int, default=5128)
    parser.add_argument("--answer-judge-max-retries", type=int, default=3)
    parser.add_argument("--answer-judge-retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--answer-judge-cache-dir",
        default=None,
        help=(
            "Optional cache directory for the judge SQLite cache. "
            "Defaults to <output-dir>/llm_cache/answer_judge."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_file = Path(args.prediction_file).expanduser().resolve()
    if not prediction_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else prediction_file.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    total_query_count, total_query_count_source = infer_total_query_count(
        args,
        prediction_file,
    )
    if total_query_count is None:
        print("[WARN] Using prediction row count as the final metric denominator.")
    else:
        print(
            "Using total query count "
            f"{total_query_count} from {total_query_count_source}."
        )

    evaluator_cls = EVALUATORS[args.dataset]
    evaluator = evaluator_cls(
        prediction_file=str(prediction_file),
        output_dir=str(output_dir),
        seed=args.seed,
        enable_llm_answer_judge=not args.disable_llm_answer_judge,
        answer_judge_model=args.answer_judge_model,
        answer_judge_base_url=args.answer_judge_base_url,
        answer_judge_api_key_env=args.answer_judge_api_key_env,
        answer_judge_max_workers=args.answer_judge_max_workers,
        answer_judge_max_tokens=args.answer_judge_max_tokens,
        answer_judge_max_retries=args.answer_judge_max_retries,
        answer_judge_retry_sleep=args.answer_judge_retry_sleep,
        answer_judge_cache_dir=args.answer_judge_cache_dir,
        total_query_count=total_query_count,
    )
    metrics = evaluator.evaluate()

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics to: {metrics_path}")
    if not args.disable_llm_answer_judge:
        print(f"Saved judge details to: {output_dir / 'llm_answer_judge.jsonl'}")


if __name__ == "__main__":
    main()
