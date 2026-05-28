# Batch inference for QA on the test set.
N_GPU=${N_GPU:-4}
DATA_ROOT=${DATA_ROOT:-"data"}
TASK_NAME=${TASK_NAME:-"hotpotqa"} # hotpotqa musique 2wikimultihopqa literaryqa
DATA_NAME=${DATA_NAME:-"${TASK_NAME}_test"}
LLM=${LLM:-"gpt-4o-mini"}
DOC_TOP_K=${DOC_TOP_K:-5}
N_THREAD=${N_THREAD:-10}
EXTRA_ARGS=${EXTRA_ARGS:-""}
torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.stage3_qa_inference \
    dataset.root=${DATA_ROOT} \
    qa_prompt=${TASK_NAME} \
    qa_evaluator=${TASK_NAME} \
    llm.model_name_or_path=${LLM} \
    test.n_threads=${N_THREAD} \
    test.top_k=${DOC_TOP_K} \
    dataset.data_name=${DATA_NAME} \
    ${EXTRA_ARGS}
