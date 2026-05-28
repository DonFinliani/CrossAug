#!/usr/bin/env python3
"""Aggregate per-book LiteraryQA GFM-RAG metrics.

GFM-RAG runs LiteraryQA as isolated per-book stage1 datasets, so stage3 writes
one metrics.json per book/variant. This helper reads the latest metrics for
each per-book data_name and reports query-weighted averages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-output-root",
        type=Path,
        default=Path("outputs") / "qa_inference",
        help="Root containing GFM-RAG stage3 output directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "literaryqa_aggregate_metrics.json",
        help="Where to save the aggregate summary.",
    )
    return parser.parse_args()


def latest_metrics_for_data_name(qa_output_root: Path, data_name: str) -> dict[str, Any] | None:
    data_dir = qa_output_root / data_name
    if not data_dir.exists():
        return None
    candidates = sorted(
        data_dir.glob("**/metrics.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    with candidates[0].open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics["_metrics_path"] = str(candidates[0])
    return metrics


def iter_variant_metrics(qa_output_root: Path, variant: str) -> list[dict[str, Any]]:
    prefix = f"literaryqa_{variant}__"
    metrics: list[dict[str, Any]] = []
    for data_dir in sorted(qa_output_root.glob(f"{prefix}*")):
        if not data_dir.is_dir():
            continue
        item = latest_metrics_for_data_name(qa_output_root, data_dir.name)
        if item is None:
            continue
        item["_data_name"] = data_dir.name
        metrics.append(item)
    return metrics


def numeric_metric_items(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if not key.startswith("_")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and key not in {"num_total_queries", "num_predictions", "num_raw_predictions"}
    }


def aggregate(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = {}
    weights: dict[str, int] = {}
    total_examples = 0
    for metrics in metrics_list:
        weight = int(metrics.get("num_total_queries") or metrics.get("num_predictions") or 0)
        if weight <= 0:
            continue
        total_examples += weight
        for key, value in numeric_metric_items(metrics).items():
            totals[key] = totals.get(key, 0.0) + value * weight
            weights[key] = weights.get(key, 0) + weight
    return {
        "books": len(metrics_list),
        "examples": total_examples,
        "metrics": {
            key: round(totals[key] / weights[key], 6)
            for key in sorted(totals)
            if weights.get(key, 0) > 0
        },
        "inputs": [
            {
                "data_name": metrics.get("_data_name"),
                "metrics_path": metrics.get("_metrics_path"),
                "num_total_queries": metrics.get("num_total_queries"),
                "num_predictions": metrics.get("num_predictions"),
            }
            for metrics in metrics_list
        ],
    }


def main() -> None:
    args = parse_args()
    qa_output_root = args.qa_output_root.expanduser().resolve()
    summary = {
        "qa_output_root": str(qa_output_root),
        "base": aggregate(iter_variant_metrics(qa_output_root, "base")),
        "gnn": aggregate(iter_variant_metrics(qa_output_root, "gnn")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved LiteraryQA aggregate metrics to: {args.output}")


if __name__ == "__main__":
    main()
