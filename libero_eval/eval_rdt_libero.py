import os
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0, help="Task ID in libero_10 (0-9)")
    parser.add_argument("--num-traj", type=int, default=25, help="Number of trajectories to test")
    parser.add_argument("--pretrained-path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--dataset-name", type=str, default="libero_10", 
                        choices=["libero_10", "libero_90"], help="Dataset name")
    return parser.parse_args()


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
    
    # 1. 加载模型
    print("Loading model...")
    config_path = 'configs/base.yaml'
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    
    pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    
    policy = create_model(
        args=config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_path,
        pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path
    )
    
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
    
    # 3. 加载语言嵌入
    text_embed = load_language_embedding(task_name, args.dataset_name, policy)
    
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
    success_count = 0
    
    base_seed = 20241201
    import tqdm
    
    for episode in tqdm.trange(total_episodes):
        # 使用不同的初始状态
        init_state_id = episode % len(init_states)
        
        env.seed(episode + base_seed)
        obs = env.reset()
        env.set_init_state(init_states[init_state_id])
        
        policy.reset()

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

        # 获取 proprio 状态
        joint_states = obs['robot0_joint_pos']
        gripper_states = obs['robot0_gripper_qpos']
        proprio = torch.from_numpy(
            np.concatenate([joint_states, gripper_states], axis=-1)
        ).float()

        global_steps = 0
        done = False
        task_success = False

        # 🎯 重新规划频率：建议从1开始测试
        REPLAN_FREQ = 1  # 每1步重新预测（推荐从这个开始）
        
        while global_steps < MAX_EPISODE_STEPS and not done:
            # 准备图像输入
            image_arrs = []
            for i in range(2):  # img_history_size = 2
                image_arrs.append(agentview_window[i])      # 外部相机
                image_arrs.append(eye_in_hand_window[i])    # 右手腕
                image_arrs.append(None)                     # 左手腕（LIBERO 没有）
            
            images = [Image.fromarray(arr) if arr is not None else None
                    for arr in image_arrs]
            
            # 预测动作序列
            actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()
            
            # 调试信息
            if episode == 0 and global_steps == 0:
                print(f"\n{'='*60}")
                print(f"【首次预测调试信息】")
                print(f"  Proprio shape: {proprio.shape}, range: [{proprio.min():.4f}, {proprio.max():.4f}]")
                print(f"  Actions shape: {actions.shape}")
                print(f"  EEF vel range: [{actions[:, :6].min():.4f}, {actions[:, :6].max():.4f}]")
                print(f"  Gripper values (first 5): {actions[:5, -1]}")
                print(f"  Expected: gripper in {{-1, 1}}, EEF vel in [-1, 1]")
                print(f"{'='*60}\n")
            
            # 只执行前N步
            num_exec_steps = min(REPLAN_FREQ, actions.shape[0], MAX_EPISODE_STEPS - global_steps)
            
            for idx in range(num_exec_steps):
                action = actions[idx]
                
                # 安全检查
                if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                    print(f"⚠️  Invalid action detected at step {global_steps}, skipping...")
                    break
                
                obs, reward, done, info = env.step(action)
                
                # 更新观察
                agentview_window.append(obs['agentview_image'])
                eye_in_hand_window.append(obs['robot0_eye_in_hand_image'])
                
                # 更新 proprio
                joint_states = obs['robot0_joint_pos']
                gripper_states = obs['robot0_gripper_qpos']
                proprio = torch.from_numpy(
                    np.concatenate([joint_states, gripper_states], axis=-1)
                ).float()
                
                global_steps += 1
                
                # 进度监控（仅第一个episode）
                if episode == 0 and global_steps % 50 == 0:
                    print(f"  → Step {global_steps:3d}: reward={reward:.2f}")
                
                if done:
                    task_success = (reward > 0)
                    break
            
            # 如果任务完成，跳出外层循环
            if done:
                break
        
        # 循环外更新成功计数
        if task_success:
            success_count += 1
        
        # 增强的进度输出
        status = "✓ SUCCESS" if task_success else "✗ FAILED"
        print(f"Trial {episode+1:3d}/{total_episodes}: {status} | reward={reward:.2f} | steps={global_steps:3d}")
    
    env.close()
    
    # 7. 输出结果
    success_rate = success_count / total_episodes * 100
    print(f"\n{'='*50}")
    print(f"Task: {task_name}")
    print(f"Instruction: {instruction}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Success Count: {success_count}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()