#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

N_GPU="${N_GPU:-2}"
DATA_ROOT="${DATA_ROOT:-data/hipporag_stage1_exports}"
CHECKPOINT="${CHECKPOINT:-rmanluo/GFM-RAG-8M}"
LLM="${LLM:-Models/Qwen3-32B}"
DOC_TOP_K="${DOC_TOP_K:-5}"
N_THREAD="${N_THREAD:-32}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:9500/v1}"

stage1_ready() {
  local data_name="$1"
  [[ -f "${DATA_ROOT}/${data_name}/processed/stage1/kg.txt" ]]
}

BASE_ITEMS=(
  "hotpotqa hotpotqa_base"
  "hotpotqa hotpotqa_gnn"
  "2wikimultihopqa 2wikimultihopqa_base"
  "2wikimultihopqa 2wikimultihopqa_gnn"
  "musique musique_base"
  "musique musique_gnn"
)

VALID_NAMES=()
RUN_ITEMS=()
for item in "${BASE_ITEMS[@]}"; do
  set -- ${item}
  task_name="$1"
  data_name="$2"
  if stage1_ready "${data_name}"; then
    VALID_NAMES+=("${data_name}")
    RUN_ITEMS+=("${task_name} ${data_name}")
  else
    echo "[WARN] Skip missing stage1 data: ${data_name}" >&2
  fi
done

LITERARY_PAIRS=0
LITERARY_BASE_ONLY=0
for base_dir in "${DATA_ROOT}"/literaryqa_base__*; do
  [[ -d "${base_dir}" ]] || continue
  base_name="$(basename "${base_dir}")"
  suffix="${base_name#literaryqa_base__}"
  base_name="literaryqa_base__${suffix}"
  gnn_name="literaryqa_gnn__${suffix}"
  if stage1_ready "${base_name}"; then
    VALID_NAMES+=("${base_name}")
    RUN_ITEMS+=("literaryqa ${base_name}")
    if stage1_ready "${gnn_name}"; then
      VALID_NAMES+=("${gnn_name}")
      RUN_ITEMS+=("literaryqa ${gnn_name}")
      LITERARY_PAIRS=$((LITERARY_PAIRS + 1))
    else
      LITERARY_BASE_ONLY=$((LITERARY_BASE_ONLY + 1))
      echo "[WARN] LiteraryQA has no GNN stage1 data; will run base only: ${base_name}" >&2
    fi
  else
    echo "[WARN] Skip LiteraryQA base with missing stage1 data: ${base_name}" >&2
  fi
done

if [[ "${#VALID_NAMES[@]}" -eq 0 ]]; then
  echo "[ERROR] No valid stage1 datasets found under ${DATA_ROOT}" >&2
  exit 1
fi

VALID_NAMES_CSV="$(IFS=,; echo "${VALID_NAMES[*]}")"
echo "[INFO] Stage2 valid datasets: ${#VALID_NAMES[@]} total; LiteraryQA pairs: ${LITERARY_PAIRS}; LiteraryQA base-only: ${LITERARY_BASE_ONLY}"

torchrun --nproc_per_node="${N_GPU}" -m gfmrag.workflow.stage2_qa_finetune \
  train.checkpoint="${CHECKPOINT}" \
  datasets.cfgs.root="${DATA_ROOT}" \
  datasets.cfgs.force_rebuild=True \
  datasets.train_names='[]' \
  datasets.valid_names="[${VALID_NAMES_CSV}]" \
  train.num_epoch=0 \
  train.do_eval=False

for item in "${RUN_ITEMS[@]}"; do
  set -- ${item}
  TASK_NAME="$1"
  DATA_NAME="$2"
  echo "[INFO] Running stage3: task=${TASK_NAME}, data=${DATA_NAME}"

  torchrun --nproc_per_node="${N_GPU}" -m gfmrag.workflow.stage3_qa_inference \
    dataset.root="${DATA_ROOT}" \
    dataset.data_name="${DATA_NAME}" \
    qa_prompt="${TASK_NAME}" \
    qa_evaluator="${TASK_NAME}" \
    graph_retriever.model_path="${CHECKPOINT}" \
    llm.model_name_or_path="${LLM}" \
    llm.base_url="${OPENAI_BASE_URL}" \
    test.n_threads="${N_THREAD}" \
    test.top_k="${DOC_TOP_K}"
done

if [[ "${LITERARY_PAIRS}" -gt 0 || "${LITERARY_BASE_ONLY}" -gt 0 ]]; then
  "${PYTHON_BIN:-python}" scripts/summarize_literaryqa_metrics.py
fi
