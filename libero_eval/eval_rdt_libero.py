import os
import csv
import random

import numpy as np
import sys
import torch
import yaml
import argparse
from collections import deque
from PIL import Image

LIBERO_REPO_ROOT = "/home/zhukefei/chensiqi/LIBERO"

if LIBERO_REPO_ROOT not in sys.path:
    sys.path.insert(0, LIBERO_REPO_ROOT)

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv
from libero_rdt_model import create_model, RoboticDiffusionTransformerModel
from libero.libero.utils.video_utils import VideoWriter 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0, help="Task ID in libero_10 (0-9)")
    parser.add_argument("--num-traj", type=int, default=25, help="Number of trajectories to test")
    parser.add_argument("--pretrained-path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--dataset-name", type=str, default="libero_10", 
                        choices=["libero_10", "libero_90"], help="Dataset name")
    # 添加视频参数
    parser.add_argument("--save-videos", action="store_true", help="Save evaluation videos")
    parser.add_argument("--video-dir", type=str, default="outs/videos", help="Directory to save videos")
    # 添加LoRA参数
    parser.add_argument("--lora-weights", type=str, default=None, 
                        help="Path to LoRA weights (if using LoRA fine-tuned model)")
    parser.add_argument(
    "--metrics-path",
    type=str,
    default=None,
    help="评估结果保存的 CSV 路径；如果提供，则每个 task 追加一行"
    )
    args = parser.parse_args()
     # ====== 任务范围约束：根据 dataset_name 检查 task-id ======
    if args.dataset_name == "libero_10":
        if not (0 <= args.task_id < 10):
            parser.error("For dataset 'libero_10', --task-id must be in [0, 9].")
    elif args.dataset_name == "libero_90":
        if not (0 <= args.task_id < 90):
            parser.error("For dataset 'libero_90', --task-id must be in [0, 89].")

    return args

def set_global_seeds(seed: int):
    """
    统一控制 Python / NumPy / PyTorch 的随机种子，便于复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_task_name_from_bddl(bddl_file_path):
    """从 BDDL 文件路径提取任务名称"""
    filename = os.path.basename(bddl_file_path)
    return filename.replace('.bddl', '')


def extract_instruction_from_task_name(task_name):
    """从任务名称提取指令"""
    if task_name[0].isupper():
        scene_pos = task_name.find("SCENE")
        if scene_pos != -1:
            if "SCENE10" in task_name:
                language_part = task_name[scene_pos + 8:]
            else:
                language_part = task_name[scene_pos + 7:]
            return language_part.replace('_', ' ')
    return task_name.replace('_', ' ')


def load_language_embedding(task_name, dataset_name="libero_10", policy=None):
    """
    加载语言嵌入（优先使用预计算，否则动态编码）
    """
    lang_embed_path = os.path.join("outs/libero_embeddings", dataset_name, f"{task_name}.pt")
    
    if os.path.exists(lang_embed_path):
        print(f"✓ Loading pre-computed embedding: {task_name}")
        lang_data = torch.load(lang_embed_path)
        # 提取 embeddings 键
        if isinstance(lang_data, dict):
            embeddings = lang_data['embeddings']
        else:
            embeddings = lang_data
        
        # 确保有 batch 维度 [B, seq_len, hidden_dim]
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)  # [seq_len, hidden_dim] -> [1, seq_len, hidden_dim]
        
        return embeddings
    else:
        if policy is None:
            raise ValueError(f"Language embedding not found and no policy provided")
        
        print(f"⚠ Embedding not found, encoding on-the-fly: {task_name}")
        instruction = extract_instruction_from_task_name(task_name)
        return policy.encode_instruction(instruction)


def main():
    args = parse_args()
    
    # 1. 加载模型与配置
    print("Loading model...")
    config_path = 'configs/base.yaml'
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    
    # 统一 device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")
    
    # 基础随机种子（每个 episode 会在此基础上加偏移）
    base_seed = 20241201
    set_global_seeds(base_seed)
    
    pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    
    policy = create_model(
        args=config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_path,
        pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
    )
    # Note: policy.reset() is already called in __init__ which handles device placement and eval mode
    
    # 2. 获取任务信息
    benchmark_dict = get_benchmark_dict()
    task_suite = benchmark_dict[args.dataset_name]()
    task = task_suite.get_task(args.task_id)
    
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    
    task_name = get_task_name_from_bddl(bddl_file)
    instruction = extract_instruction_from_task_name(task_name)
    
    print(f"Task ID: {args.task_id}")
    print(f"Task Name: {task_name}")
    print(f"Instruction: {instruction}")
    
    # 3. 加载语言嵌入，并搬到 device / dtype
    text_embed = load_language_embedding(task_name, args.dataset_name, policy)
    text_embed = text_embed.to(device=device, dtype=torch.bfloat16)
    
    # 4. 创建环境
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128
    )
    
    # 5. 获取初始状态
    init_states = task_suite.get_task_init_states(args.task_id)
    
    # 6. 评估循环
    MAX_EPISODE_STEPS = 400
    total_episodes = args.num_traj
    success_count = 0   # libero 风格：有 done=True 的 episode 数
    
    import tqdm
    from collections import deque
    
    # 创建视频保存目录
    video_folder = os.path.join(
        args.video_dir,
        f"{args.dataset_name}_task{args.task_id}"
    )
    
    # 使用 VideoWriter 上下文管理器
    with VideoWriter(video_folder, save_video=args.save_videos, fps=30, single_video=False) as video_writer:
        for episode in tqdm.trange(total_episodes):
            # 使用不同的初始状态
            init_state_id = episode % len(init_states)
            
            # 为每个 episode 设置独立但可复现的 seed
            episode_seed = base_seed + episode
            env.seed(episode_seed)
            set_global_seeds(episode_seed)
            
            obs = env.reset()
            env.set_init_state(init_states[init_state_id])
            
            policy.reset()
            video_writer.reset()  # 重置视频缓冲
            
            # 维护两个图像历史窗口
            agentview_window = deque(maxlen=2)
            eye_in_hand_window = deque(maxlen=2)
            
            # 获取初始图像
            agentview_img = obs['agentview_image']
            eye_in_hand_img = obs['robot0_eye_in_hand_image']
            
            # 用第一帧填充历史（与训练一致）
            for _ in range(2):
                agentview_window.append(agentview_img)
                eye_in_hand_window.append(eye_in_hand_img)
            
            # 获取 proprio 状态（joint + gripper），并搬到 device / dtype
            joint_states = obs['robot0_joint_pos']
            gripper_states = obs['robot0_gripper_qpos']
            proprio_np = np.concatenate([joint_states, gripper_states], axis=-1).astype(np.float32)
            proprio = torch.from_numpy(proprio_np).to(device=device, dtype=torch.bfloat16)
            
            global_steps = 0
            done = False
            episode_done = False   # libero 风格：这一条 episode 是否出现过 done=True
            task_success = False   # 使用 info['success'] 判定“真正完成任务”
            reward = 0.0
            info = {}
            
            # 🎯 重新规划频率（一次预测多少步）
            REPLAN_FREQ = 8
            
            while global_steps < MAX_EPISODE_STEPS and not done:
                # 准备图像输入
                image_arrs = []
                for i in range(2):  # img_history_size = 2
                    image_arrs.append(agentview_window[i])      # 外部相机
                    image_arrs.append(eye_in_hand_window[i])    # 右手腕
                    image_arrs.append(None)                     # 左手腕（LIBERO 没有）
                
                images = [Image.fromarray(arr) if arr is not None else None
                          for arr in image_arrs]
                
                # 预测动作序列（推理模式，不构建计算图）
                with torch.inference_mode():
                    actions = policy.step(proprio, images, text_embed).squeeze(0)
                actions = actions.detach().cpu().numpy()
                
                # 调试信息（首帧）
                if episode == 0 and global_steps == 0:
                    print(f"\n{'='*60}")
                    print(f"【首次预测调试信息】")
                    print(f"  Proprio shape: {proprio.shape}, range: [{proprio.min().item():.4f}, {proprio.max().item():.4f}]")
                    print(f"  Actions shape: {actions.shape}")
                    print(f"  EEF vel range: [{actions[:, :6].min():.4f}, {actions[:, :6].max():.4f}]")
                    print(f"  Gripper values (first 5): {actions[:5, -1]}")
                    print(f"  Expected: gripper in {{-1, 1}}, EEF vel in [-1, 1]")
                    print(f"{'='*60}\n")
                
                # 只执行前 N 步
                num_exec_steps = min(REPLAN_FREQ, actions.shape[0], MAX_EPISODE_STEPS - global_steps)
                
                for idx in range(num_exec_steps):
                    action = actions[idx]
                    
                    # 安全检查
                    if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                        print(f"⚠️  Invalid action detected at step {global_steps}, skipping...")
                        break
                    
                    obs, reward, done, info = env.step(action)
                    
                    # 记录视频帧
                    video_writer.append_obs(
                        obs, 
                        done, 
                        idx=episode,
                        camera_name="agentview_image"
                    )
                    
                    # 更新观察窗口
                    agentview_window.append(obs['agentview_image'])
                    eye_in_hand_window.append(obs['robot0_eye_in_hand_image'])
                    
                    # 更新 proprio
                    joint_states = obs['robot0_joint_pos']
                    gripper_states = obs['robot0_gripper_qpos']
                    proprio_np = np.concatenate([joint_states, gripper_states], axis=-1).astype(np.float32)
                    proprio = torch.from_numpy(proprio_np).to(device=device, dtype=torch.bfloat16)
                    
                    global_steps += 1
                    
                    # 进度监控（仅第一个 episode）
                    if episode == 0 and global_steps % 50 == 0:
                        print(f"  → Step {global_steps:3d}: reward={reward:.2f}")
                    
                    # 只要 env 返回 done=True，就认为这一条 episode 结束
                    if done:
                        episode_done = True
                        break
                
                if done:
                    break
            
            # 循环外更新 “libero 风格成功计数”：这一条 episode 是否终止
            if episode_done:
                success_count += 1
            
            # 增强的进度输出：status 看的是 info['success']
            status = "✓ SUCCESS" if episode_done else "✗ FAILED"
            print(
                f"Trial {episode+1:3d}/{total_episodes}: {status} "
                f"| done={episode_done} info['success']={info.get('success', False)} "
                f"| steps={global_steps:3d}"
            )
        
        # VideoWriter 会在退出 with 块时自动保存所有视频
    
    env.close()
    
    # 7. 输出结果（success_rate 为 0~1）
    success_rate = success_count / total_episodes
    print(f"\n{'='*50}")
    print(f"Task: {task_name}")
    print(f"Instruction: {instruction}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Episode Done Count (libero-style): {success_count}")
    print(f"Success Rate (libero-style): {success_rate * 100:.2f}%")
    print(f"{'='*50}")
    
    if args.save_videos:
        print(f"\n📹 Videos saved to: {video_folder}")
    
    # 8. 写入 CSV（如果提供了 --metrics-path）
    if getattr(args, "metrics_path", None) is not None:
        metrics_path = args.metrics_path
        
        # 确保目录存在（可能只有文件名，没有目录）
        metrics_dir = os.path.dirname(metrics_path)
        if metrics_dir != "":
            os.makedirs(metrics_dir, exist_ok=True)
        
        file_exists = os.path.isfile(metrics_path) and os.path.getsize(metrics_path) > 0
        
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            # 第一次写入时写表头
            if not file_exists:
                writer.writerow([
                    "dataset_name",      # 如 libero_90
                    "task_id",           # 任务 id
                    "task_name",         # 解析后的任务名
                    "instruction",       # 语言描述
                    "num_traj",          # episode 数
                    "episode_done_count",# 有 done 的 episode 数
                    "success_rate",      # 成功率（0~1，libero-style）
                    "checkpoint_path",   # 模型路径
                    "video_dir",         # 视频目录
                ])
            
            writer.writerow([
                args.dataset_name,
                args.task_id,
                task_name,
                instruction,
                total_episodes,
                success_count,
                success_rate,
                args.pretrained_path,
                video_folder if args.save_videos else "",
            ])
        
        print(f"📄 Metrics appended to CSV: {metrics_path}")


if __name__ == "__main__":
    main()