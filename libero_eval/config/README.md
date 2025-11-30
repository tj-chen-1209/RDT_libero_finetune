# RDT-LIBERO 评估配置说明

## 快速开始

### 1️⃣ 编辑配置文件
```bash
vim libero_eval/config/eval_config.yaml
```

修改以下关键配置：
- `dataset.name`: 选择要评估的数据集
- `model.pretrained_path`: 你的模型checkpoint路径
- `model.lora_adapter`: 如果使用LoRA，填写adapter路径
- `evaluation.num_traj`: 每个任务评估的次数

### 2️⃣ 运行评估
```bash
./eval_all_tasks_fast.sh
```

就这么简单！

## 配置文件结构

```yaml
dataset:
  name: libero_spatial           # 数据集名称

model:
  pretrained_path: ./checkpoints/xxx/checkpoint-20000   # 模型路径
  lora_adapter: ./checkpoints/xxx/checkpoint-20000      # LoRA路径（可选）

evaluation:
  num_traj: 3                    # 每任务评估次数
  seed: 20241201                 # 随机种子
  video:
    save_interval: 0             # 视频保存间隔（0=不保存）
    root_dir: outs/eval_videos   # 视频保存目录

output:
  metrics_dir: outs/metrics      # 结果保存目录
  prefix: "FAST_"                # 文件名前缀
```

## 常用场景

### 快速测试（3 episodes）
```yaml
evaluation:
  num_traj: 3
  video:
    save_interval: 0
output:
  prefix: "FAST_"
```

### 标准评估（10 episodes）
```yaml
evaluation:
  num_traj: 10
  video:
    save_interval: 10
output:
  prefix: "STD_"
```

### 完整评估（20 episodes）
```yaml
evaluation:
  num_traj: 20
  video:
    save_interval: 10
output:
  prefix: "FULL_"
```

## 数据集选项

- `libero_10`: 10个基础任务
- `libero_90`: 90个任务（需要较长时间）
- `libero_spatial`: 10个空间任务
- `libero_object`: 10个物体操作任务
- `libero_goal`: 10个目标任务

## 高级用法

### 使用自定义配置文件
```bash
# 创建新配置
cp libero_eval/config/eval_config.yaml my_config.yaml

# 使用自定义配置运行
./eval_all_tasks_fast.sh my_config.yaml
```

### 不使用 LoRA
如果不使用 LoRA 微调，在配置文件中设置：
```yaml
model:
  lora_adapter: null
```

或直接注释掉该行。

## 输出说明

评估完成后，结果会保存在 `outs/metrics/` 目录下，文件名格式：
```
{prefix}{dataset_name}_eval{num_traj}eps_{checkpoint}_seed{seed}_{timestamp}.csv
```

例如：
```
FAST_libero_spatial_eval3eps_checkpoint-20000_seed20241201_20251129_215130.csv
```

CSV 文件包含：
- 每个任务的成功率
- 详细的任务信息
- 平均成功率（在最后一行）

## 故障排查

### 配置文件不存在
确保 `libero_eval/config/eval_config.yaml` 文件存在。

### Checkpoint 路径错误
检查配置文件中的路径是否正确：
```bash
ls -l ./checkpoints/your-checkpoint/
```

### 环境未激活
脚本会自动激活 `rdt_libero_eval` 环境，如果失败，手动激活：
```bash
conda activate rdt_libero_eval
```

