# #!/bin/bash

# # 评估脚本 - 带视频录制功能
# # 使用方法: bash eval_with_video.sh

# cd /home/zhukefei/chensiqi/RDT_libero_finetune
# run_id=$(date +%Y%m%d_%H%M%S)
# # 激活 conda 环境
# source /share_data/zhukefei/miniconda3/etc/profile.d/conda.sh
# conda activate rdt_libero_eval

# # ========== 配置参数 ==========
# TASK_ID=1                  # 任务 ID (libero_10: 0-9)
# NUM_TRAJ=3                 # 评估轨迹数量（建议测试时用3个）
# DATASET_NAME="libero_10"   # 数据集名称
# CHECKPOINT="./checkpoints/rdt-finetune-1b-20251119_122234/checkpoint-30000"
# # CHECKPOINT="./checkpoints/rdt-1b"
# CHECKPOINT_NAME=$(basename ${CHECKPOINT})
# echo ""
# echo "========== 完整评估（录制视频）=========="
# python libero_eval/eval_rdt_libero.py \
#     --task-id ${TASK_ID} \
#     --num-traj ${NUM_TRAJ} \
#     --pretrained-path ${CHECKPOINT} \
#     --dataset-name ${DATASET_NAME} \
#     --save-videos \
#     --video-dir outs/eval_videos/${DATASET_NAME}_task${TASK_ID}_${CHECKPOINT_NAME}_${run_id}

# echo ""
# echo "✅ 评估完成！"
# echo "📹 视频保存位置: outs/eval_videos/${DATASET_NAME}_task${TASK_ID}_${CHECKPOINT_NAME}_${run_id}/"


#!/bin/bash
cd /home/zhukefei/chensiqi/RDT_libero_finetune
run_id=$(date +%Y%m%d_%H%M%S)
source /share_data/zhukefei/miniconda3/etc/profile.d/conda.sh
conda activate rdt_libero_eval

# DATASET_NAME="libero_90"
DATASET_NAME="libero_10"
NUM_TRAJ=20   # 正式评估 测试为3
CHECKPOINT="./checkpoints/rdt-finetune-1b-20251119_122234/checkpoint-45000"
CHECKPOINT_NAME=$(basename "${CHECKPOINT}")

# 所有 task 共用一个 CSV
METRICS_PATH="outs/metrics/${DATASET_NAME}_${CHECKPOINT_NAME}_${run_id}.csv"

if [ "$DATASET_NAME" == "libero_10" ]; then
    TASK_RANGE=$(seq 0 9)
elif [ "$DATASET_NAME" == "libero_90" ]; then
    TASK_RANGE=$(seq 0 89)
else
    echo "Invalid dataset name: ${DATASET_NAME}"
    exit 1
fi

for TASK_ID in ${TASK_RANGE}; do
    VIDEO_DIR="outs/eval_videos/${DATASET_NAME}_${CHECKPOINT_NAME}_${run_id}/task${TASK_ID}"
    echo ""
    echo "==== Task ${TASK_ID} ===="
    python libero_eval/eval_rdt_libero.py \
        --task-id ${TASK_ID} \
        --num-traj ${NUM_TRAJ} \
        --pretrained-path ${CHECKPOINT} \
        --dataset-name ${DATASET_NAME} \
        --save-videos \
        --video-dir "${VIDEO_DIR}" \
        --metrics-path "${METRICS_PATH}"
done
