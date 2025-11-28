export run_id=$(date +%Y%m%d_%H%M%S)
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0
export DS_BUILD_EVOFORMER_ATTN=0
# export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES=0


export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"

#========================================================================
# 训练配置：数据集和微调方法
#========================================================================
dataset_name="libero_spatial"      # 数据集: libero_10, libero_spatial, libero_object, libero_goal, libero_90
finetune_method="LoRA"             # 微调方法: LoRA (参数高效) 或 Full (全参数)
model_size="1B"                    # 模型大小: 1B
lora_rank=32                       # LoRA rank
lora_alpha=64                      # LoRA alpha
seed=42                            # 随机种子（用于可复现性）

# 基础模型路径（如果从checkpoint继续训练）
base_model_name="spatial-ckpt20k"  # 基础模型标识：scratch(从头), spatial-ckpt20k(从spatial 20k步继续)
BASE_MODEL_PATH="./checkpoints/rdt-rdt_libero_sft_csq-libero_spatial-20251125_114702/checkpoint-20000"

export WANDB_PROJECT="rdt_libero_sft_lora_csq"

#========================================================================
# LoRA 微调模式（推荐用于快速实验和资源受限场景）
# ========================================================================
# 优势：
#   - 显存占用少（约节省50%）
#   - 训练速度快（快1.5-2倍）
#   - 权重文件小（几MB vs 几GB）
#   - 便于版本管理和分享
#========================================================================

# 生成清晰的输出文件夹名称
# 格式: RDT-{model_size}-{method}-{dataset}-from_{base_model}-r{rank}a{alpha}-{timestamp}
if [ -z "$BASE_MODEL_PATH" ]; then
    # 从头训练
    export LORA_OUTPUT_DIR="./checkpoints/RDT-${model_size}-${finetune_method}-${dataset_name}-r${lora_rank}a${lora_alpha}-${run_id}"
else
    # 从checkpoint继续训练
    export LORA_OUTPUT_DIR="./checkpoints/RDT-${model_size}-${finetune_method}-${dataset_name}-from_${base_model_name}-r${lora_rank}a${lora_alpha}-${run_id}"
fi

#========================================================================
# 打印训练配置
#========================================================================
echo "=========================================================================="
echo "🚀 RDT LoRA 微调训练"
echo "=========================================================================="
echo "📊 数据集:        ${dataset_name}"
echo "🔧 微调方法:      ${finetune_method}"
echo "📦 模型大小:      ${model_size}"
if [ -n "$BASE_MODEL_PATH" ]; then
    echo "🎯 基础模型:      ${base_model_name}"
    echo "📂 模型路径:      ${BASE_MODEL_PATH}"
else
    echo "🎯 基础模型:      从头训练 (scratch)"
fi
echo "🎯 LoRA Rank:     ${lora_rank}"
echo "🎯 LoRA Alpha:    ${lora_alpha}"
echo "🌱 随机种子:      ${seed}"
echo "📁 输出目录:      ${LORA_OUTPUT_DIR}"
echo "⏰ 运行时间戳:    ${run_id}"
echo "=========================================================================="

if [ ! -d "$LORA_OUTPUT_DIR" ]; then
    mkdir -p "$LORA_OUTPUT_DIR"
    echo "✅ 输出文件夹已创建: '$LORA_OUTPUT_DIR'"
else
    echo "⚠️  输出文件夹已存在: '$LORA_OUTPUT_DIR'"
fi

# 保存训练配置到输出目录
cat > "$LORA_OUTPUT_DIR/training_config.txt" << EOF
训练配置信息
=====================================
数据集:          ${dataset_name}
微调方法:        ${finetune_method}
模型大小:        ${model_size}
基础模型:        ${base_model_name:-scratch}
基础模型路径:    ${BASE_MODEL_PATH:-N/A}
LoRA Rank:       ${lora_rank}
LoRA Alpha:      ${lora_alpha}
LoRA Dropout:    0.1
随机种子:        ${seed}
训练批次大小:    32
学习率:          1e-4
混合精度:        bf16
最大训练步数:    200000
运行时间戳:      ${run_id}
=====================================
EOF

echo ""
echo "🔄 开始训练..."
echo ""

# 构建 deepspeed 命令
if [ -n "$BASE_MODEL_PATH" ]; then
    # 从checkpoint继续训练
    echo "📥 加载基础模型: ${BASE_MODEL_PATH}"
    deepspeed --exclude="localhost:0" main_sft.py \
        --deepspeed="./configs/zero2.json" \
        --pretrained_model_name_or_path=$BASE_MODEL_PATH \
        --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
        --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
        --output_dir=$LORA_OUTPUT_DIR \
        --seed=${seed} \
        --use_lora \
        --lora_rank=${lora_rank} \
        --lora_alpha=${lora_alpha} \
        --lora_dropout=0.1 \
        --lora_target_modules="all" \
        --train_batch_size=32 \
        --sample_batch_size=32 \
        --num_sample_batches=4 \
        --max_train_steps=200000 \
        --checkpointing_period=5000 \
        --sample_period=500 \
        --checkpoints_total_limit=40 \
        --lr_scheduler="constant" \
        --learning_rate=1e-4 \
        --mixed_precision="bf16" \
        --dataloader_num_workers=8 \
        --image_aug \
        --dataset_type="finetune" \
        --state_noise_snr=40 \
        --load_from_hdf5 \
        --report_to=tensorboard \
        --precomp_lang_embed
else
    # 从头训练
    echo "🆕 从头开始训练"
    deepspeed --exclude="localhost:0" main_sft.py \
        --deepspeed="./configs/zero2.json" \
        --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
        --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
        --output_dir=$LORA_OUTPUT_DIR \
        --seed=${seed} \
        --use_lora \
        --lora_rank=${lora_rank} \
        --lora_alpha=${lora_alpha} \
        --lora_dropout=0.1 \
        --lora_target_modules="all" \
        --train_batch_size=32 \
        --sample_batch_size=32 \
        --num_sample_batches=4 \
        --max_train_steps=200000 \
        --checkpointing_period=5000 \
        --sample_period=500 \
        --checkpoints_total_limit=40 \
        --lr_scheduler="constant" \
        --learning_rate=1e-4 \
        --mixed_precision="bf16" \
        --dataloader_num_workers=8 \
        --image_aug \
        --dataset_type="finetune" \
        --state_noise_snr=40 \
        --load_from_hdf5 \
        --report_to=tensorboard \
        --precomp_lang_embed
fi

