#!/bin/bash
#========================================================================
# 快速验证脚本 - 基于 finetune_lora_sft.sh
# 用途：运行10步验证配置是否正确
#========================================================================

export run_id=$(date +%Y%m%d_%H%M%S)
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=WARN  # 减少日志输出
export NCCL_NVLS_ENABLE=0
export DS_BUILD_EVOFORMER_ATTN=0
export CUDA_VISIBLE_DEVICES=1  # 只用一张卡验证

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"

#========================================================================
# 训练配置（与 finetune_lora_sft.sh 一致）
#========================================================================
dataset_name="libero_spatial"
finetune_method="LoRA"
model_size="1B"
lora_rank=32
lora_alpha=64
seed=42

base_model_name="spatial-ckpt20k"
BASE_MODEL_PATH="./checkpoints/rdt-rdt_libero_sft_csq-libero_spatial-20251125_114702/checkpoint-20000"

export WANDB_PROJECT="rdt_libero_sft_lora_csq"

# 验证输出目录
export VERIFY_DIR="./checkpoints/VERIFY-${dataset_name}-${run_id}"

#========================================================================
# 开始验证
#========================================================================
echo "=========================================================================="
echo "🔍 快速验证 (10步)"
echo "=========================================================================="
echo "📊 数据集: ${dataset_name}"
echo "🎯 基础模型: ${base_model_name}"
echo "🎯 LoRA: r${lora_rank}/a${lora_alpha}"
echo "📁 验证输出: ${VERIFY_DIR}"
echo "=========================================================================="
echo ""

# 运行10步验证
python main_sft.py \
    --pretrained_model_name_or_path=$BASE_MODEL_PATH \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$VERIFY_DIR \
    --seed=${seed} \
    --use_lora \
    --lora_rank=${lora_rank} \
    --lora_alpha=${lora_alpha} \
    --lora_dropout=0.1 \
    --lora_target_modules="all" \
    --train_batch_size=2 \
    --sample_batch_size=2 \
    --num_sample_batches=1 \
    --max_train_steps=10 \
    --checkpointing_period=10000 \
    --sample_period=5 \
    --checkpoints_total_limit=1 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=2 \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=tensorboard \
    --precomp_lang_embed

EXIT_CODE=$?

echo ""
echo "=========================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 验证成功！可以开始正式训练"
    echo ""
    echo "启动命令: bash finetune_lora_sft.sh"
    echo "清理验证: rm -rf $VERIFY_DIR"
else
    echo "❌ 验证失败！退出码: $EXIT_CODE"
fi
echo "=========================================================================="

exit $EXIT_CODE

