# Build the index for testing dataset
N_GPU=1
DATA_ROOT="data"
DATA_NAME_LIST="hotpotqa_test 2wikimultihopqa_test musique_test"
COLBERT_CKPT="/home/GDM/cache/huggingface/colbertv2.0-local"
export TORCH_EXTENSIONS_DIR="/home/GDM/cache/torch_extensions"
for DATA_NAME in ${DATA_NAME_LIST}; do
   python -m gfmrag.workflow.stage1_index_dataset \
   dataset.root=${DATA_ROOT} \
   dataset.data_name=${DATA_NAME} \
   el_model.model_name_or_path=${COLBERT_CKPT}
done


# Build the index for training dataset

# N_GPU=1
# DATA_ROOT="data"
# DATA_NAME_LIST="hotpotqa_train musique_train 2wikimultihopqa_train" #
# START_N=0
# END_N=19
# for i in $(seq ${START_N} ${END_N}); do
#    for DATA_NAME in ${DATA_NAME_LIST}; do
#       DATA_NAME=${DATA_NAME}${i}
#       python -m gfmrag.workflow.stage1_index_dataset \
#       dataset.root=${DATA_ROOT} \
#       dataset.data_name=${DATA_NAME}
#    done
# done
