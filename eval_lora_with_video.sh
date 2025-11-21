#!/bin/bash

# LoRA 模型评估脚本 - 带视频录制功能
# 使用方法: bash eval_lora_with_video.sh

cd /home/zhukefei/chensiqi/RDT_libero_finetune

# 激活评估环境
source /share_data/zhukefei/miniconda3/etc/profile.d/conda.sh
conda activate rdt_libero_eval

# ========== 配置参数 ==========
TASK_ID=0                  # 任务 ID (libero_10: 0-9)
NUM_TRAJ=25                # 评估轨迹数量
DATASET_NAME="libero_10"   # 数据集名称

# LoRA 模型配置
BASE_MODEL="./checkpoints/rdt-1b"  # 基础模型路径
LORA_WEIGHTS="./checkpoints/rdt-finetune-1b-lora-XXXXXX/checkpoint-20000"  # LoRA权重路径（需要修改）

# 检查LoRA权重路径是否存在
if [ ! -d "$LORA_WEIGHTS" ]; then
    echo "❌ 错误：LoRA权重路径不存在: ${LORA_WEIGHTS}"
    echo "请修改脚本中的 LORA_WEIGHTS 变量，指向实际的LoRA checkpoint目录"
    exit 1
fi

echo "=========================================="
echo "LoRA 模型评估"
echo "=========================================="
echo "基础模型: ${BASE_MODEL}"
echo "LoRA权重: ${LORA_WEIGHTS}"
echo "任务: ${DATASET_NAME}, Task ${TASK_ID}"
echo "轨迹数量: ${NUM_TRAJ}"
echo "=========================================="

# 运行评估（带视频录制）
python libero_eval/eval_rdt_libero.py \
    --task-id ${TASK_ID} \
    --num-traj ${NUM_TRAJ} \
    --pretrained-path ${BASE_MODEL} \
    --lora-weights ${LORA_WEIGHTS} \
    --dataset-name ${DATASET_NAME} \
    --save-videos \
    --video-dir outs/eval_videos_lora

echo ""
echo "✅ LoRA模型评估完成！"
echo "📹 视频保存位置: outs/eval_videos_lora/${DATASET_NAME}_task${TASK_ID}/"

