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
dataset_name="libero_spatial"
RESUME_CHECKPOINT_SRC="./checkpoints/rdt-finetune-1b-20251119_122234/checkpoint-65000"

export WANDB_PROJECT="rdt_libero_sft_csq"
BASE_OUTPUT_DIR="./checkpoints/rdt-${WANDB_PROJECT}-${dataset_name}"
export OUTPUT_DIR="./checkpoints/rdt-${WANDB_PROJECT}-${dataset_name}-${run_id}" # 加入run_id作为后缀
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
# export CUTLASS_PATH="/path/to/cutlass"



if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

RESUME_ARG=""
if [ -n "$RESUME_CHECKPOINT_SRC" ]; then
    if [ ! -d "$RESUME_CHECKPOINT_SRC" ]; then
        echo "指定的恢复路径 '$RESUME_CHECKPOINT_SRC' 不存在，终止执行。"
        exit 1
    fi
    resume_basename=$(basename "$RESUME_CHECKPOINT_SRC")
    if [ ! -d "$OUTPUT_DIR/$resume_basename" ]; then
        echo "拷贝历史 checkpoint 到当前输出目录: $resume_basename"
        rsync -a "$RESUME_CHECKPOINT_SRC" "$OUTPUT_DIR/"
    fi
    resume_abs_path=$(realpath "$OUTPUT_DIR/$resume_basename")
    RESUME_ARG="--pretrained_model_name_or_path=$resume_abs_path"
fi

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     test
#    --max_train_steps=200000 \
#    --checkpointing_period=10000 \
#    --sample_period=500 \
# --pretrained_model_name_or_path="./checkpoints/rdt-1b" \


    # --pretrained_model_name_or_path="./checkpoints/rdt-1b" \
# --use_8bit_adam \
# deepspeed main.py \
# deepspeed --hostfile=hostfile.txt main_sft.py \
deepspeed --exclude="localhost:0" main_sft.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=32 \
    --sample_batch_size=64 \
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
    --report_to=wandb \
    --precomp_lang_embed \
    $RESUME_ARG
    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training
    # --precomp_lang_embed \

# ========================================================================
# LoRA 微调模式（推荐用于快速实验和资源受限场景）
# ========================================================================
# 优势：
#   - 显存占用少（约节省50%）
#   - 训练速度快（快1.5-2倍）
#   - 权重文件小（几MB vs 几GB）
#   - 便于版本管理和分享
# 
# 使用方法：取消下面的注释，注释掉上面的全参数微调命令
# ========================================================================
# export LORA_OUTPUT_DIR="${BASE_OUTPUT_DIR}-lora-${run_id}"
# 
# if [ ! -d "$LORA_OUTPUT_DIR" ]; then
#     mkdir "$LORA_OUTPUT_DIR"
#     echo "LoRA output folder '$LORA_OUTPUT_DIR' created"
# fi
# 
# deepspeed --exclude="localhost:0" main.py \
#     --deepspeed="./configs/zero2.json" \
#     --pretrained_model_name_or_path="./checkpoints/rdt-1b" \
#     --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
#     --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
#     --output_dir=$LORA_OUTPUT_DIR \
#     --use_lora \
#     --lora_rank=32 \
#     --lora_alpha=64 \
#     --lora_dropout=0.1 \
#     --lora_target_modules="all" \
#     --train_batch_size=64 \
#     --sample_batch_size=128 \
#     --max_train_steps=200000 \
#     --checkpointing_period=5000 \
#     --sample_period=500 \
#     --checkpoints_total_limit=40 \
#     --lr_scheduler="constant" \
#     --learning_rate=1e-4 \
#     --mixed_precision="bf16" \
#     --dataloader_num_workers=8 \
#     --image_aug \
#     --dataset_type="finetune" \
#     --state_noise_snr=40 \
#     --load_from_hdf5 \
#     --report_to=wandb \
#     --use_8bit_adam
#     # --precomp_lang_embed  # 使用预先计算的语言嵌入（如果有的话）
