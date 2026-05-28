#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 2 ]]; then
  cat >&2 <<'USAGE'
Usage:
  bash scripts/evaluate_only.bash <prediction.jsonl> <dataset> [output_dir]

Datasets:
  hotpotqa | musique | 2wikimultihopqa | literaryqa

Environment overrides:
  ANSWER_JUDGE_MODEL=deepseek-v4-pro
  ANSWER_JUDGE_BASE_URL=https://api.deepseek.com
  ANSWER_JUDGE_API_KEY_ENV=DEEPSEEK_API_KEY
  ANSWER_JUDGE_MAX_WORKERS=100
  ANSWER_JUDGE_MAX_TOKENS=5128
  ANSWER_JUDGE_MAX_RETRIES=3
  ANSWER_JUDGE_RETRY_SLEEP=2.0
  TOTAL_QUERY_COUNT=<optional denominator>
  TEST_FILE=<optional test.json/jsonl>
  DATASET_ROOT=<optional root containing data_name/processed/stage1/test.json>
  DATA_NAME=<optional stage1 data name>
  NO_AUTO_TOTAL_QUERY_COUNT=true
  PYTHON_BIN=python
  DISABLE_LLM_ANSWER_JUDGE=true
USAGE
  exit 2
fi

PREDICTION_FILE="$1"
DATASET="$2"
OUTPUT_DIR="${3:-$(dirname "${PREDICTION_FILE}")}"

EXTRA_ARGS=()
if [[ -n "${TOTAL_QUERY_COUNT:-}" ]]; then
  EXTRA_ARGS+=(--total-query-count "${TOTAL_QUERY_COUNT}")
fi
if [[ -n "${TEST_FILE:-}" ]]; then
  EXTRA_ARGS+=(--test-file "${TEST_FILE}")
fi
if [[ -n "${DATASET_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${DATA_NAME:-}" ]]; then
  EXTRA_ARGS+=(--data-name "${DATA_NAME}")
fi
if [[ "${NO_AUTO_TOTAL_QUERY_COUNT:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--no-auto-total-query-count)
fi
if [[ "${DISABLE_LLM_ANSWER_JUDGE:-true}" == "true" ]]; then
  EXTRA_ARGS+=(--disable-llm-answer-judge)
else
  EXTRA_ARGS+=(--enable-llm-answer-judge)
fi

"${PYTHON_BIN:-python}" scripts/evaluate_predictions.py \
  --prediction-file "${PREDICTION_FILE}" \
  --dataset "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --answer-judge-model "${ANSWER_JUDGE_MODEL:-deepseek-v4-pro}" \
  --answer-judge-base-url "${ANSWER_JUDGE_BASE_URL:-https://api.deepseek.com}" \
  --answer-judge-api-key-env "${ANSWER_JUDGE_API_KEY_ENV:-DEEPSEEK_API_KEY}" \
  --answer-judge-max-workers "${ANSWER_JUDGE_MAX_WORKERS:-100}" \
  --answer-judge-max-tokens "${ANSWER_JUDGE_MAX_TOKENS:-5128}" \
  --answer-judge-max-retries "${ANSWER_JUDGE_MAX_RETRIES:-3}" \
  --answer-judge-retry-sleep "${ANSWER_JUDGE_RETRY_SLEEP:-2.0}" \
  "${EXTRA_ARGS[@]}"
