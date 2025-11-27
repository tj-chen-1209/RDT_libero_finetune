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
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero_rdt_model import create_model, RoboticDiffusionTransformerModel
from libero.libero.utils.video_utils import VideoWriter 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0, help="Task ID in libero_10 (0-9)")
    parser.add_argument("--num-traj", type=int, default=25, help="Number of trajectories to test")
    parser.add_argument("--pretrained-path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--dataset-name", type=str, default="libero_10", 
                        choices=["libero_10", "libero_90","libero_spatial","libero_goal","libero_object"], help="Dataset name")
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

def _get_obs_item(obs, key, idx):
    """
    兼容两种可能的 vector obs 结构：
    - obs 是 dict[np.ndarray] -> obs[key][idx]
    - obs 是 list[dict]       -> obs[idx][key]
    """
    if isinstance(obs, dict):
        return obs[key][idx]
    else:
        return obs[idx][key]

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
    # lang_embed_path = os.path.join("outs/libero_embeddings", dataset_name, f"{task_name}")

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
    
    # 4. 获取初始状态（LIBERO benchmark 自带）
    init_states = task_suite.get_task_init_states(args.task_id)
    num_init_states = init_states.shape[0]

    # 5. 创建并行环境（SubprocVectorEnv）
    env_num = args.num_traj  # 并行的 env 数 = 一次评估的 episode 数
    env_args = dict(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
    )

    env = SubprocVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
    )
    env.seed(base_seed)
    env.reset()

    # 给每个 env 分配一个 init_state（循环使用）
    indices = np.arange(env_num) % num_init_states
    init_states_batch = init_states[indices]
    obs = env.set_init_state(init_states_batch)

    # 6. 为每个 env 准备图像历史 & proprio
    MAX_EPISODE_STEPS = 400
    dones = [False] * env_num
    global_steps = 0

    from collections import deque

    agentview_windows = [deque(maxlen=2) for _ in range(env_num)]
    eye_in_hand_windows = [deque(maxlen=2) for _ in range(env_num)]
    proprios = [None for _ in range(env_num)]

    for i in range(env_num):
        agent_img = _get_obs_item(obs, "agentview_image", i)
        eye_img = _get_obs_item(obs, "robot0_eye_in_hand_image", i)

        for _ in range(2):
            agentview_windows[i].append(agent_img)
            eye_in_hand_windows[i].append(eye_img)

        joint_states = _get_obs_item(obs, "robot0_joint_pos", i)
        gripper_states = _get_obs_item(obs, "robot0_gripper_qpos", i)
        proprio_np = np.concatenate([joint_states, gripper_states], axis=-1).astype(np.float32)
        proprios[i] = torch.from_numpy(proprio_np).to(device=device, dtype=torch.bfloat16)
        
    # 物理预热：用零动作空跑几步
    for _ in range(5):
        env.step(np.zeros((env_num, 7), dtype=np.float32))


    # 创建视频保存目录
    video_folder = os.path.join(
        args.video_dir,
        f"{args.dataset_name}_task{args.task_id}"
    )
    os.makedirs(video_folder, exist_ok=True)

    num_success = 0  # 有 done=True 的 env 数（libero 风格）

    with VideoWriter(video_folder, save_video=args.save_videos, fps=30, single_video=True) as video_writer:
        while (global_steps < MAX_EPISODE_STEPS) and (not all(dones)):
            actions = np.zeros((env_num, 7), dtype=np.float32)

            # 逐 env 调 policy.step，构成 (env_num, 7) 的 action
            with torch.inference_mode():
                for i in range(env_num):
                    if dones[i]:
                        # 已经结束的 env 扔 0 动作
                        continue

                    # 准备该 env 的历史图像
                    image_arrs = []
                    for t in range(2):  # img_history_size = 2
                        image_arrs.append(agentview_windows[i][t])       # 外部相机
                        image_arrs.append(eye_in_hand_windows[i][t])     # 右手腕
                        image_arrs.append(None)                          # 左手腕（LIBERO 没有）

                    images = [
                        Image.fromarray(arr) if arr is not None else None
                        for arr in image_arrs
                    ]

                    # RDT 推理：返回 [H, action_dim]，只取第 0 步
                    action_seq = policy.step(proprios[i], images, text_embed).squeeze(0)
                    action_seq = action_seq.detach().cpu().numpy()

                    if np.any(np.isnan(action_seq)) or np.any(np.isinf(action_seq)):
                        print(f"⚠️ Invalid action at env {i}, step {global_steps}, use zero.")
                        continue

                    actions[i] = action_seq[0]  # 只用第一步动作

            # 首次调试信息（看 env 0）
            if global_steps == 0:
                p0 = proprios[0]
                print(f"\n{'='*60}")
                print("【首次预测调试信息】(env 0)")
                print(f"  Proprio shape: {p0.shape}, range: [{p0.min().item():.4f}, {p0.max().item():.4f}]")
                print(f"  Actions shape: {actions.shape}")
                print(f"  EEF vel range: [{actions[:, :6].min():.4f}, {actions[:, :6].max():.4f}]")
                print(f"  Gripper values (first 5 envs): {actions[:5, -1]}")
                print("  Expected: gripper in {-1, 1}, EEF vel in [-1, 1]")
                print(f"{'='*60}\n")

            # 并行 step
            obs, reward, done, info = env.step(actions)
            global_steps += 1

            # 记录视频：现在用 vector 接口
            video_writer.append_vector_obs(
                obs, dones, camera_name="agentview_image"
            )

            # 更新每个 env 的状态
            for i in range(env_num):
                # 更新 done（libero 风格：累积 or）
                dones[i] = dones[i] or bool(done[i])

                # 只更新还没结束的 env 的观测与 proprio
                if not dones[i]:
                    agent_img = _get_obs_item(obs, "agentview_image", i)
                    eye_img = _get_obs_item(obs, "robot0_eye_in_hand_image", i)
                    agentview_windows[i].append(agent_img)
                    eye_in_hand_windows[i].append(eye_img)

                    joint_states = _get_obs_item(obs, "robot0_joint_pos", i)
                    gripper_states = _get_obs_item(obs, "robot0_gripper_qpos", i)
                    proprio_np = np.concatenate([joint_states, gripper_states], axis=-1).astype(np.float32)
                    proprios[i] = torch.from_numpy(proprio_np).to(device=device, dtype=torch.bfloat16)

        # rollout 结束后，统计 success
        for i in range(env_num):
            num_success += int(dones[i])
    
    env.close()

    # 7. 输出结果（env_num = args.num_traj）
    success_rate = num_success / float(env_num)
    print(f"\n{'='*50}")
    print(f"Task: {task_name}")
    print(f"Instruction: {instruction}")
    print(f"Num Parallel Envs (= Episodes): {env_num}")
    print(f"Episode Done Count (libero-style): {num_success}")
    print(f"Success Rate (libero-style): {success_rate * 100:.2f}%")
    print(f"Max Steps Per Episode: {MAX_EPISODE_STEPS}")
    print(f"{'='*50}")

    if args.save_videos:
        print(f"\n📹 Videos saved to: {video_folder}")

    if getattr(args, "metrics_path", None) is not None:
        metrics_path = args.metrics_path
        metrics_dir = os.path.dirname(metrics_path)
        if metrics_dir != "":
            os.makedirs(metrics_dir, exist_ok=True)

        file_exists = os.path.isfile(metrics_path) and os.path.getsize(metrics_path) > 0

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "dataset_name",
                    "task_id",
                    "task_name",
                    "instruction",
                    "num_traj",             # 并行 env 数
                    "episode_done_count",   # done=True 的 env 数
                    "success_rate",         # 0~1
                    "checkpoint_path",
                    "video_dir",
                ])

            writer.writerow([
                args.dataset_name,
                args.task_id,
                task_name,
                instruction,
                env_num,
                num_success,
                success_rate,
                args.pretrained_path,
                video_folder if args.save_videos else "",
            ])

        print(f"📄 Metrics appended to CSV: {metrics_path}")



if __name__ == "__main__":
    main()