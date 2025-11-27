#!/bin/bash
cd /home/zhukefei/chensiqi/RDT_libero_finetune
run_id=$(date +%Y%m%d_%H%M%S)
source /share_data/zhukefei/miniconda3/etc/profile.d/conda.sh
conda activate rdt_libero_eval

# DATASET_NAME="libero_90"
DATASET_NAME="libero_spatial"

# 并行 env 数 (= 每个 task 的 episode 数)
# 调试：可以设成 3；正式评估：建议 20 跟 LIBERO 一致
NUM_ENVS=1

CHECKPOINT="./checkpoints/rdt-rdt_libero_sft_csq-libero_spatial-20251125_114702/checkpoint-65000"
CHECKPOINT_NAME=$(basename "${CHECKPOINT}")

# 所有 task 共用一个 CSV
METRICS_PATH="outs/metrics/${DATASET_NAME}_${CHECKPOINT_NAME}_${run_id}.csv"

# 根据数据集决定 task 范围
if [ "$DATASET_NAME" == "libero_10" ]; then
    TASK_RANGE=$(seq 0 9)
elif [ "$DATASET_NAME" == "libero_90" ]; then
    TASK_RANGE=$(seq 0 89)
elif [ "$DATASET_NAME" == "libero_goal" ]; then
    TASK_RANGE=$(seq 0 9)
elif [ "$DATASET_NAME" == "libero_object" ]; then
    TASK_RANGE=$(seq 0 9)
elif [ "$DATASET_NAME" == "libero_spatial" ]; then
    TASK_RANGE=$(seq 0 9)
else
    echo "Invalid dataset name: ${DATASET_NAME}"
    exit 1
fi

for TASK_ID in ${TASK_RANGE}; do
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

    python libero_eval/eval_rdt_libero_subEnv.py \
        --task-id ${TASK_ID} \
        --num-traj ${NUM_ENVS} \
        --pretrained-path ${CHECKPOINT} \
        --dataset-name ${DATASET_NAME} \
        ${SAVE_VIDEOS_FLAG} \
        ${VIDEO_DIR_ARG} \
        --metrics-path "${METRICS_PATH}"
done

# ====== 计算平均成功率 ======
echo ""
echo "=========================================="
echo "计算所有 task 的平均成功率..."
echo "=========================================="

python3 << EOF
import csv
import os

metrics_path = "${METRICS_PATH}"

if not os.path.exists(metrics_path):
    print(f"错误: CSV 文件不存在: {metrics_path}")
    exit(1)

success_rates = []
task_count = 0

with open(metrics_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            success_rate = float(row['success_rate'])
            success_rates.append(success_rate)
            task_count += 1
        except (ValueError, KeyError) as e:
            print(f"警告: 跳过无效行: {row}")
            continue

if task_count == 0:
    print("错误: 没有找到有效的成功率数据")
    exit(1)

avg_success_rate = sum(success_rates) / len(success_rates)
total_success = sum(success_rates)
total_episodes = sum([int(row['episode_done_count']) for row in csv.DictReader(open(metrics_path))])

print(f"\n{'='*60}")
print(f"评估结果汇总")
print(f"{'='*60}")
print(f"数据集: ${DATASET_NAME}")
print(f"Checkpoint: ${CHECKPOINT}")
print(f"总 Task 数: {task_count}")
print(f"平均成功率: {avg_success_rate * 100:.2f}%")
print(f"总成功 Episode 数: {total_success:.0f}")
print(f"总 Episode 数: {task_count * ${NUM_ENVS}}")
print(f"{'='*60}\n")

# 将平均值追加到 CSV 文件
with open(metrics_path, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "${DATASET_NAME}",
        "AVG",
        "AVERAGE",
        "Average success rate across all tasks",
        task_count * ${NUM_ENVS},
        int(total_success),
        avg_success_rate,
        "${CHECKPOINT}",
        ""
    ])

print(f"平均值已追加到 CSV: {metrics_path}")
EOF

echo ""
echo "评估完成！"

