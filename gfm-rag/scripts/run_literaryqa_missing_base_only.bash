#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

N_GPU="${N_GPU:-2}"
DATA_ROOT="${DATA_ROOT:-data/hipporag_stage1_exports}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/qa_inference}"
CHECKPOINT="${CHECKPOINT:-rmanluo/GFM-RAG-8M}"
LLM="${LLM:-Models/Qwen3-32B}"
DOC_TOP_K="${DOC_TOP_K:-5}"
N_THREAD="${N_THREAD:-32}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:9500/v1}"
RUN_STAGE2="${RUN_STAGE2:-true}"
SKIP_EXISTING_METRICS="${SKIP_EXISTING_METRICS:-true}"
DRY_RUN="${DRY_RUN:-false}"

MISSING_BOOK_INDICES="${MISSING_BOOK_INDICES:-0005 0008 0017 0031 0032 0036 0038 0042}"

stage1_ready() {
  local data_name="$1"
  [[ -f "${DATA_ROOT}/${data_name}/processed/stage1/kg.txt" ]]
}

has_metrics() {
  local data_name="$1"
  compgen -G "${OUTPUT_ROOT}/${data_name}"'/*/*/metrics.json' >/dev/null
}

run_or_echo() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

find_base_data_name() {
  local book_idx="$1"
  local matches=("${DATA_ROOT}"/literaryqa_base__literaryqa_test_"${book_idx}"_*)

  if [[ "${#matches[@]}" -eq 0 || ! -d "${matches[0]}" ]]; then
    echo "[ERROR] Missing LiteraryQA base stage1 directory for book index ${book_idx}" >&2
    return 1
  fi
  if [[ "${#matches[@]}" -gt 1 ]]; then
    echo "[ERROR] Multiple LiteraryQA base stage1 directories for book index ${book_idx}:" >&2
    printf '  %s\n' "${matches[@]}" >&2
    return 1
  fi

  basename "${matches[0]}"
}

VALID_NAMES=()
RUN_ITEMS=()

for book_idx in ${MISSING_BOOK_INDICES}; do
  data_name="$(find_base_data_name "${book_idx}")"

  if ! stage1_ready "${data_name}"; then
    echo "[ERROR] Stage1 data exists but kg.txt is missing: ${DATA_ROOT}/${data_name}/processed/stage1/kg.txt" >&2
    exit 1
  fi

  if [[ "${SKIP_EXISTING_METRICS}" == "true" ]] && has_metrics "${data_name}"; then
    echo "[INFO] Skip existing metrics: ${data_name}"
    continue
  fi

  VALID_NAMES+=("${data_name}")
  RUN_ITEMS+=("literaryqa ${data_name}")
done

if [[ "${#RUN_ITEMS[@]}" -eq 0 ]]; then
  echo "[INFO] No missing LiteraryQA base runs left. Set SKIP_EXISTING_METRICS=false to rerun."
  exit 0
fi

VALID_NAMES_CSV="$(IFS=,; echo "${VALID_NAMES[*]}")"
echo "[INFO] LiteraryQA missing-base datasets to run: ${#RUN_ITEMS[@]}"
printf '[INFO]   %s\n' "${VALID_NAMES[@]}"

if [[ "${RUN_STAGE2}" == "true" ]]; then
  stage2_cmd=(torchrun --nproc_per_node="${N_GPU}" -m gfmrag.workflow.stage2_qa_finetune \
    train.checkpoint="${CHECKPOINT}" \
    datasets.cfgs.root="${DATA_ROOT}" \
    datasets.cfgs.force_rebuild=True \
    datasets.train_names='[]' \
    datasets.valid_names="[${VALID_NAMES_CSV}]" \
    train.num_epoch=0 \
    train.do_eval=False)
  run_or_echo "${stage2_cmd[@]}"
else
  echo "[INFO] RUN_STAGE2=false, skip stage2 preparation."
fi

for item in "${RUN_ITEMS[@]}"; do
  set -- ${item}
  TASK_NAME="$1"
  DATA_NAME="$2"
  echo "[INFO] Running stage3: task=${TASK_NAME}, data=${DATA_NAME}"

  stage3_cmd=(torchrun --nproc_per_node="${N_GPU}" -m gfmrag.workflow.stage3_qa_inference \
    dataset.root="${DATA_ROOT}" \
    dataset.data_name="${DATA_NAME}" \
    qa_prompt="${TASK_NAME}" \
    qa_evaluator="${TASK_NAME}" \
    graph_retriever.model_path="${CHECKPOINT}" \
    llm.model_name_or_path="${LLM}" \
    llm.base_url="${OPENAI_BASE_URL}" \
    test.n_threads="${N_THREAD}" \
    test.top_k="${DOC_TOP_K}")
  run_or_echo "${stage3_cmd[@]}"
done

echo "[INFO] Done. Re-run metrics summary with:"
echo "  python3 scripts/summarize_literaryqa_metrics.py"
