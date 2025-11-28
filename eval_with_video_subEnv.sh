#!/bin/bash

# ============================================================================
# RDT-LIBERO 评估脚本 (支持并行环境 + 视频保存)
# ============================================================================
# 功能：
#   1. 遍历指定数据集的所有任务，逐个评估
#   2. 支持并行环境加速评估
#   3. 可选择性保存视频（默认每10个任务保存一次）
#   4. 自动计算并记录平均成功率
#   5. 支持随机种子，保证可复现性
# ============================================================================

set -e  # 遇到错误立即退出

# ----------------------------------------------------------------------------
# 1. 环境配置
# ----------------------------------------------------------------------------
cd /home/zhukefei/chensiqi/RDT_libero_finetune

# 激活 conda 环境
source /share_data/zhukefei/miniconda3/etc/profile.d/conda.sh
conda activate rdt_libero_eval

# 生成运行 ID（用于区分不同次评估）
RUN_ID=$(date +%Y%m%d_%H%M%S)

echo "============================================================================"
echo "开始评估 - 运行 ID: ${RUN_ID}"
echo "============================================================================"
echo ""

# ----------------------------------------------------------------------------
# 2. 评估参数配置
# ----------------------------------------------------------------------------

# 数据集选择
DATASET_NAME="libero_spatial"
# 可选: libero_10, libero_90, libero_spatial, libero_object, libero_goal

# Checkpoint 路径配置
# 如果是 LoRA checkpoint，需要同时指定 BASE_MODEL 和 LORA_ADAPTER
# 如果是完整 checkpoint，只需要指定 CHECKPOINT，将 BASE_MODEL 留空

# 方式1: 评估完整 checkpoint（非 LoRA）
# CHECKPOINT="./checkpoints/rdt-rdt_libero_sft_csq-libero_spatial-20251125_114702/checkpoint-20000"
# BASE_MODEL=""
# LORA_ADAPTER=""

# 方式2: 评估 LoRA checkpoint（需要指定 base model）
BASE_MODEL="./checkpoints/rdt-rdt_libero_sft_csq-libero_spatial-20251125_114702/checkpoint-20000"
LORA_ADAPTER="./checkpoints/RDT-1B-LoRA-libero_spatial-from_spatial-ckpt20k-r32a64-20251127_235342/checkpoint-5000"

# 根据配置确定使用哪个路径
if [ -n "$LORA_ADAPTER" ]; then
    # LoRA 模式：使用 LoRA adapter 名称作为标识
    CHECKPOINT="$LORA_ADAPTER"
    CHECKPOINT_NAME=$(basename "${LORA_ADAPTER}")
else
    # 完整 checkpoint 模式
    CHECKPOINT_NAME=$(basename "${CHECKPOINT}")
fi

# 评估轮次（每个任务的 episode 数量）
NUM_EPISODES=20
# 建议: 调试时设为 1-3; 正式评估时设为 20（与 LIBERO 官方一致）

# 随机种子（保证可复现性）
RANDOM_SEED=20241201

# 视频保存间隔（每隔多少个任务保存一次视频，0 表示不保存）
VIDEO_SAVE_INTERVAL=10
# 例如: 设为 10 时，只保存 task 0, 10, 20, 30... 的视频

echo "配置信息:"
echo "  数据集: ${DATASET_NAME}"
echo "  Checkpoint: ${CHECKPOINT_NAME}"
echo "  每任务评估轮次: ${NUM_EPISODES}"
echo "  随机种子: ${RANDOM_SEED}"
echo "  视频保存间隔: ${VIDEO_SAVE_INTERVAL} (0=不保存)"
echo ""

# ----------------------------------------------------------------------------
# 3. 输出路径配置
# ----------------------------------------------------------------------------

# 评估指标保存路径（所有任务共用一个 CSV 文件）
# 文件名格式: {数据集}_eval{轮次}eps_{checkpoint}_seed{种子}_{时间戳}.csv
# 示例: libero_spatial_eval20eps_checkpoint-5000_seed20241201_20251128_135446.csv
METRICS_PATH="outs/metrics/${DATASET_NAME}_eval${NUM_EPISODES}eps_${CHECKPOINT_NAME}_seed${RANDOM_SEED}_${RUN_ID}.csv"
mkdir -p "$(dirname "${METRICS_PATH}")"

# 视频保存根目录
# 目录名格式: {数据集}_eval{轮次}eps_{checkpoint}_seed{种子}_{时间戳}
VIDEO_ROOT_DIR="outs/eval_videos/${DATASET_NAME}_eval${NUM_EPISODES}eps_${CHECKPOINT_NAME}_seed${RANDOM_SEED}_${RUN_ID}"

echo "输出路径:"
echo "  📊 评估结果CSV: ${METRICS_PATH}"
echo "  🎥 视频保存目录: ${VIDEO_ROOT_DIR}"
echo ""

# ----------------------------------------------------------------------------
# 4. 确定任务范围
# ----------------------------------------------------------------------------

# 根据数据集确定任务 ID 范围
case "${DATASET_NAME}" in
    libero_10)
        TASK_RANGE=$(seq 0 9)
        TOTAL_TASKS=10
        ;;
    libero_90)
        TASK_RANGE=$(seq 0 89)
        TOTAL_TASKS=90
        ;;
    libero_spatial|libero_object|libero_goal)
        TASK_RANGE=$(seq 0 9)
        TOTAL_TASKS=10
        ;;
    *)
        echo "错误: 不支持的数据集名称 '${DATASET_NAME}'"
        echo "支持的数据集: libero_10, libero_90, libero_spatial, libero_object, libero_goal"
        exit 1
        ;;
esac

echo "任务范围: 0 到 $((TOTAL_TASKS - 1))，共 ${TOTAL_TASKS} 个任务"
echo ""

# ----------------------------------------------------------------------------
# 5. 开始逐任务评估
# ----------------------------------------------------------------------------

echo "============================================================================"
echo "开始评估所有任务..."
echo "============================================================================"
echo ""

TASK_COUNT=0

for TASK_ID in ${TASK_RANGE}; do
    TASK_COUNT=$((TASK_COUNT + 1))
    
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "正在评估 Task ${TASK_ID}/${TOTAL_TASKS} (进度: ${TASK_COUNT}/${TOTAL_TASKS})"
    echo "────────────────────────────────────────────────────────────────────────────"

    # 根据设置决定是否保存视频
    SAVE_VIDEOS_FLAG=""
    VIDEO_DIR_ARG=""
    
    # 如果设置了视频保存间隔，且当前任务 ID 是间隔的倍数，则保存视频
    if [ ${VIDEO_SAVE_INTERVAL} -gt 0 ] && [ $((TASK_ID % VIDEO_SAVE_INTERVAL)) -eq 0 ]; then
        VIDEO_DIR="${VIDEO_ROOT_DIR}/task${TASK_ID}"
        SAVE_VIDEOS_FLAG="--save-videos"
        VIDEO_DIR_ARG="--video-dir ${VIDEO_DIR}"
        echo "  [视频] 本任务将保存视频到: ${VIDEO_DIR}"
    fi

    # 执行评估
    # 根据是否使用 LoRA 构建不同的命令参数
    if [ -n "$LORA_ADAPTER" ]; then
        # LoRA 模式：传递 base model 和 lora adapter
        python libero_eval/eval_rdt_libero_subEnv.py \
            --task-id ${TASK_ID} \
            --num-traj ${NUM_EPISODES} \
            --pretrained-path ${BASE_MODEL} \
            --lora-adapter ${LORA_ADAPTER} \
            --dataset-name ${DATASET_NAME} \
            --seed ${RANDOM_SEED} \
            ${SAVE_VIDEOS_FLAG} \
            ${VIDEO_DIR_ARG} \
            --metrics-path "${METRICS_PATH}"
    else
        # 完整 checkpoint 模式
        python libero_eval/eval_rdt_libero_subEnv.py \
            --task-id ${TASK_ID} \
            --num-traj ${NUM_EPISODES} \
            --pretrained-path ${CHECKPOINT} \
            --dataset-name ${DATASET_NAME} \
            --seed ${RANDOM_SEED} \
            ${SAVE_VIDEOS_FLAG} \
            ${VIDEO_DIR_ARG} \
            --metrics-path "${METRICS_PATH}"
    fi
    
    echo ""
done

# ----------------------------------------------------------------------------
# 6. 计算并保存统计结果
# ----------------------------------------------------------------------------

echo "============================================================================"
echo "所有任务评估完成！正在计算统计结果..."
echo "============================================================================"
echo ""

python3 << 'PYTHON_SCRIPT'
import csv
import os
import sys

# 从环境变量读取参数
metrics_path = "${METRICS_PATH}"
dataset_name = "${DATASET_NAME}"
checkpoint = "${CHECKPOINT}"
num_episodes = ${NUM_EPISODES}
random_seed = ${RANDOM_SEED}
run_id = "${RUN_ID}"

# 检查 CSV 文件是否存在
if not os.path.exists(metrics_path):
    print(f"❌ 错误: 评估结果文件不存在: {metrics_path}")
    sys.exit(1)

# 读取所有任务的评估结果
success_rates = []
task_ids = []
episode_success_counts = []

with open(metrics_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            success_rate = float(row['success_rate'])
            episode_done = int(row['episode_done_count'])
            
            success_rates.append(success_rate)
            task_ids.append(row['task_id'])
            episode_success_counts.append(episode_done)
        except (ValueError, KeyError) as e:
            print(f"⚠️  警告: 跳过无效数据行: {row}")
            continue

# 检查是否有有效数据
if len(success_rates) == 0:
    print("❌ 错误: 没有找到有效的评估数据")
    sys.exit(1)

# 计算统计指标
total_tasks = len(success_rates)
avg_success_rate = sum(success_rates) / total_tasks
total_success_episodes = sum(episode_success_counts)
total_episodes = total_tasks * num_episodes

# 打印详细结果
print("╔" + "═" * 78 + "╗")
print("║" + " " * 28 + "评估结果汇总" + " " * 38 + "║")
print("╠" + "═" * 78 + "╣")
print(f"║ 运行 ID          : {run_id:<58} ║")
print(f"║ 数据集           : {dataset_name:<58} ║")
print(f"║ Checkpoint       : {os.path.basename(checkpoint):<58} ║")
print(f"║ 随机种子         : {random_seed:<58} ║")
print("╠" + "═" * 78 + "╣")
print(f"║ 评估任务总数     : {total_tasks:<58} ║")
print(f"║ 每任务评估轮次   : {num_episodes:<58} ║")
print(f"║ 总评估轮次       : {total_episodes:<58} ║")
print("╠" + "═" * 78 + "╣")
print(f"║ ✓ 平均成功率     : {avg_success_rate * 100:>6.2f}%{' ' * 50} ║")
print(f"║ ✓ 成功轮次       : {total_success_episodes}/{total_episodes}{' ' * (52 - len(str(total_success_episodes)) - len(str(total_episodes)))} ║")
print("╚" + "═" * 78 + "╝")
print()

# 显示每个任务的详细结果
print("任务详情:")
print("─" * 80)
for i, (task_id, success_rate, success_count) in enumerate(zip(task_ids, success_rates, episode_success_counts)):
    status = "✓" if success_rate == 1.0 else "✗" if success_rate == 0.0 else "◐"
    print(f"  {status} Task {task_id:>2s}: {success_rate * 100:>6.2f}% ({success_count}/{num_episodes})")
print("─" * 80)
print()

# 将平均值追加到 CSV 文件
with open(metrics_path, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        dataset_name,
        "AVG",
        "AVERAGE",
        f"Average success rate across all tasks (seed={random_seed})",
        total_episodes,
        total_success_episodes,
        avg_success_rate,
        checkpoint,
        run_id
    ])

print(f"✅ 评估结果已保存到: {metrics_path}")
PYTHON_SCRIPT


# ----------------------------------------------------------------------------
# 7. 评估完成
# ----------------------------------------------------------------------------

echo ""
echo "============================================================================"
echo "✅ 评估任务全部完成！"
echo "============================================================================"
echo ""
echo "📊 结果文件: ${METRICS_PATH}"
if [ ${VIDEO_SAVE_INTERVAL} -gt 0 ]; then
    echo "🎥 视频目录: ${VIDEO_ROOT_DIR}"
fi
echo ""
echo "提示: 可以使用以下命令查看结果"
echo "  cat ${METRICS_PATH}"
echo ""
