#!/bin/bash
cd /home/zhukefei/chensiqi/RDT_libero_finetune
run_id=$(date +%Y%m%d_%H%M%S)
source /share_data/zhukefei/miniconda3/etc/profile.d/conda.sh
conda activate rdt_libero_eval

# DATASET_NAME="libero_90"
DATASET_NAME="libero_90"

# 并行 env 数 (= 每个 task 的 episode 数)
# 调试：可以设成 3；正式评估：建议 20 跟 LIBERO 一致
NUM_ENVS=20

CHECKPOINT="./checkpoints/rdt-finetune-1b-20251119_122234/checkpoint-45000"
CHECKPOINT_NAME=$(basename "${CHECKPOINT}")

# 所有 task 共用一个 CSV
METRICS_PATH="outs/metrics/${DATASET_NAME}_${CHECKPOINT_NAME}_${run_id}.csv"

# 根据数据集决定 task 范围
if [ "$DATASET_NAME" == "libero_10" ]; then
    TASK_RANGE=$(seq 0 9)
elif [ "$DATASET_NAME" == "libero_90" ]; then
    TASK_RANGE=$(seq 0 89)
else
    echo "Invalid dataset name: ${DATASET_NAME}"
    exit 1
fi

# for TASK_ID in ${TASK_RANGE}; do
for TASK_ID in {84..89}; do
    echo ""
    echo "==== Task ${TASK_ID} ===="

    # 默认：不保存视频
    SAVE_VIDEOS_FLAG=""
    VIDEO_DIR_ARG=""

    if [ $((TASK_ID % 10)) -eq 0 ]; then
        VIDEO_DIR="outs/eval_videos/${DATASET_NAME}_${CHECKPOINT_NAME}_${run_id}/task${TASK_ID}"
        SAVE_VIDEOS_FLAG="--save-videos"
        VIDEO_DIR_ARG="--video-dir ${VIDEO_DIR}"
    fi

    python libero_eval/eval_rdt_libero.py \
        --task-id ${TASK_ID} \
        --num-traj ${NUM_ENVS} \
        --pretrained-path ${CHECKPOINT} \
        --dataset-name ${DATASET_NAME} \
        ${SAVE_VIDEOS_FLAG} \
        ${VIDEO_DIR_ARG} \
        --metrics-path "${METRICS_PATH}"
done

