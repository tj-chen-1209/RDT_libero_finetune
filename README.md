# RDT LIBERO Finetune

本项目基于 **RDT (Robot Diffusion Transformer)** 模型，在 **LIBERO** 机器人操作数据集上进行微调和评估。

## 📖 项目简介

RDT 是一个用于机器人操作的多模态 Transformer 模型，结合了扩散模型和 Transformer 架构。本项目专注于在 LIBERO 基准数据集上微调 RDT 模型，支持多种训练模式（全参数微调、LoRA 微调）和完整的评估流程。

### 主要特性

- ✅ 支持 LIBERO-10、LIBERO-90、LIBERO-Spatial、LIBERO-Object、LIBERO-Goal 等多个数据集
- ✅ 支持全参数微调和 LoRA 高效微调
- ✅ 集成 DeepSpeed 进行分布式训练
- ✅ 支持多 GPU 训练和评估
- ✅ 完整的评估流程，支持视频录制
- ✅ WandB 实验跟踪
- ✅ 预计算语言嵌入加速训练

## 🛠️ 环境配置

### 1. 创建 Conda 环境

```bash
# 创建训练环境
conda env create -f environment.yml
conda activate rdt

# 安装额外依赖
pip install -r requirements.txt
```

### 2. 创建评估环境（可选）

```bash
# 评估环境需要额外的依赖（如 robosuite 等）
conda env create -f environment_libero_eval.yml
conda activate rdt_libero_eval
```

### 主要依赖

- Python 3.10
- PyTorch 2.1.0
- DeepSpeed 0.14.2
- Transformers 4.38.2+
- Diffusers 0.26.3+
- Flash-Attention 2.8.3
- WandB 0.17.0

## 📦 数据准备

### 下载 LIBERO 数据集

```bash
python download_libero.py
```

数据集将下载到 `data/datasets/` 目录下，包括：
- `libero_10` - 10 个基础任务
- `libero_90` - 90 个扩展任务
- `libero_spatial` - 空间推理任务
- `libero_object` - 物体操作任务
- `libero_goal` - 目标导向任务

### 数据预处理（可选）

如果需要预计算语言嵌入以加速训练：

```bash
# 预处理会自动在训练时进行，也可以单独预处理
python scripts/preprocess_language_embeddings.py
```

## 🚀 训练

### 全参数微调（推荐用于最佳性能）

```bash
bash finetune_sft.sh
```

主要参数说明：
- `--train_batch_size=32` - 训练批次大小
- `--learning_rate=1e-4` - 学习率
- `--max_train_steps=200000` - 最大训练步数
- `--checkpointing_period=5000` - checkpoint 保存间隔
- `--dataset_type="finetune"` - 使用微调数据集配置
- `--mixed_precision="bf16"` - 使用 BF16 混合精度
- `--load_from_hdf5` - 从 HDF5 文件加载数据
- `--precomp_lang_embed` - 使用预计算的语言嵌入

### LoRA 微调（推荐用于快速实验）

```bash
bash finetune_lora_sft.sh
```

LoRA 微调优势：
- 显存占用少（约节省 50%）
- 训练速度快（快 1.5-2 倍）
- 权重文件小（几 MB vs 几 GB）
- 便于版本管理和分享

### 分布式训练

```bash
# 多节点训练（使用 hostfile.txt 指定节点）
deepspeed --hostfile=hostfile.txt main_sft.py --deepspeed="./configs/zero2.json" ...

# 单节点多 GPU（排除 GPU 0）
deepspeed --exclude="localhost:0" main_sft.py --deepspeed="./configs/zero2.json" ...
```

### 从 Checkpoint 恢复训练

在 `finetune_sft.sh` 中设置：

```bash
RESUME_CHECKPOINT_SRC="./checkpoints/rdt-finetune-1b-20251119_122234/checkpoint-65000"
```

## 📊 评估

### 单任务评估（带视频录制）

```bash
bash eval_with_video.sh
```

配置参数：
```bash
TASK_ID=1                  # 任务 ID
NUM_TRAJ=20                # 评估轨迹数量（建议测试时用 3，正式评估用 20）
DATASET_NAME="libero_10"   # 数据集名称
CHECKPOINT="./checkpoints/rdt-finetune-1b-xxx/checkpoint-30000"
```

### 批量评估（评估所有任务）

```bash
bash eval_with_video_subEnv.sh
```

支持评估整个数据集的所有子任务，自动生成汇总 CSV 文件。

### 评估输出

评估结果保存在 `outs/` 目录：
- `outs/metrics/` - CSV 格式的评估指标
- `outs/eval_videos/` - 评估过程录制的视频
- `outs/videos/` - 训练过程生成的可视化视频

## 📁 项目结构

```
RDT_libero_finetune/
├── configs/                      # 配置文件
│   ├── base.yaml                # 基础训练配置
│   ├── zero2.json               # DeepSpeed ZeRO-2 配置
│   ├── finetune_datasets.json   # 微调数据集配置
│   └── ...
├── data/                        # 数据目录
│   └── datasets/                # LIBERO 数据集
├── models/                      # 模型定义
│   ├── rdt/                     # RDT 模型
│   ├── multimodal_encoder/      # 多模态编码器
│   └── rdt_runner.py            # 模型运行器
├── train/                       # 训练脚本
│   └── train_sft.py             # 监督微调训练逻辑
├── libero_eval/                 # 评估脚本
│   └── eval_rdt_libero.py       # LIBERO 评估脚本
├── checkpoints/                 # 模型权重（.gitignore）
├── outs/                        # 输出文件
│   ├── metrics/                 # 评估指标
│   ├── eval_videos/             # 评估视频
│   └── videos/                  # 训练视频
├── main_sft.py                  # 训练入口
├── finetune_sft.sh              # 全参数微调脚本
├── finetune_lora_sft.sh         # LoRA 微调脚本
├── eval_with_video.sh           # 单任务评估脚本
├── eval_with_video_subEnv.sh    # 批量评估脚本
├── download_libero.py           # 数据下载脚本
├── environment.yml              # Conda 环境配置（训练）
├── environment_libero_eval.yml  # Conda 环境配置（评估）
└── requirements.txt             # Python 依赖
```

## 🔧 配置说明

### 数据集配置

在 `configs/finetune_datasets.json` 中配置微调数据集：

```json
{
  "libero_10": {
    "path": "data/datasets/libero_10",
    "tasks": [...],
    "sample_weight": 1.0
  }
}
```

### 训练配置

主要配置文件 `configs/base.yaml`：

```yaml
model:
  pretrained_model_name_or_path: "./checkpoints/rdt-1b"
  vision_encoder: "google/siglip-so400m-patch14-384"
  text_encoder: "google/t5-v1_1-xxl"

training:
  batch_size: 32
  learning_rate: 1e-4
  max_steps: 200000
  ...
```

## 📈 实验追踪

本项目使用 WandB 进行实验追踪。在 `finetune_sft.sh` 中配置：

```bash
export WANDB_PROJECT="rdt_libero_sft_csq"
```

训练指标包括：
- Loss 曲线
- 学习率变化
- 采样结果可视化
- GPU 利用率

## 🎯 性能优化建议

### 训练优化
1. 使用 `--precomp_lang_embed` 预计算语言嵌入，减少重复计算
2. 使用 `--image_aug` 启用数据增强，提高泛化能力
3. 调整 `--dataloader_num_workers` 根据 CPU 核心数优化数据加载
4. 使用 DeepSpeed ZeRO-2 减少显存占用

### 评估优化
1. 测试阶段使用少量轨迹（NUM_TRAJ=3）快速验证
2. 正式评估使用足够轨迹（NUM_TRAJ=20）获得可靠结果
3. 使用 `--save-videos` 保存失败案例用于分析

## 🐛 常见问题

### 1. CUDA Out of Memory
- 减小 `train_batch_size`
- 使用 LoRA 微调模式
- 启用 DeepSpeed ZeRO-2 或 ZeRO-3
- 减少 `dataloader_num_workers`

### 2. 数据加载慢
- 确保使用 `--load_from_hdf5`
- 增加 `--dataloader_num_workers`
- 使用 SSD 存储数据集

### 3. 评估环境报错
- 确保使用独立的评估环境 `rdt_libero_eval`
- 检查 robosuite、libero 等依赖是否正确安装

### 4. DeepSpeed 错误
- 检查 `configs/zero2.json` 配置
- 确保所有节点的环境一致
- 检查网络配置（NCCL 相关环境变量）

## 📝 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{rdt2024,
  title={Robot Diffusion Transformer},
  author={...},
  journal={arXiv preprint arXiv:...},
  year={2024}
}

@inproceedings{libero2023,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={...},
  booktitle={NeurIPS},
  year={2023}
}
```

## 📄 许可证

本项目遵循 [LICENSE](LICENSE) 中指定的许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请联系：[您的邮箱]

---

**最后更新**: 2025-11-28
