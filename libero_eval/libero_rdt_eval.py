"""
优化版评估脚本：一次加载模型，评估所有任务
相比原版 eval_rdt_libero_subEnv.py，避免了重复加载模型，大幅提升评估速度
"""
import os
import csv
import random
import time

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

def load_config(config_path):
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="评估所有任务（一次加载模型）")
    
    # 配置文件参数（优先级最高）
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径（YAML格式），如果提供则其他参数可选")
    
    # 原有参数（可通过配置文件或命令行指定）
    parser.add_argument("--num-traj", type=int, default=None, help="每个任务的评估轨迹数")
    parser.add_argument("--pretrained-path", type=str, default=None, help="预训练模型路径")
    parser.add_argument("--dataset-name", type=str, default=None, 
                        choices=["libero_10", "libero_90", "libero_object", "libero_spatial", "libero_goal"], 
                        help="数据集名称")
    parser.add_argument("--lora-adapter", type=str, default=None, 
                        help="LoRA adapter 路径（如果使用 LoRA）")
    parser.add_argument("--metrics-path", type=str, default=None,
                        help="评估结果保存的 CSV 路径")
    parser.add_argument("--seed", type=int, default=None, 
                        help="随机种子（用于复现）")
    parser.add_argument("--video-save-interval", type=int, default=None,
                        help="视频保存间隔（0=不保存，10=每10个任务保存一次）")
    parser.add_argument("--video-root-dir", type=str, default=None,
                        help="视频保存根目录")
    parser.add_argument("--gpu", type=int, default=None,
                        help="指定使用的 GPU 编号（例如：0, 1, 2...），默认使用 cuda:0")
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，从配置文件加载参数
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"配置文件不存在: {args.config}")
        
        config = load_config(args.config)
        
        # 从配置文件读取参数（命令行参数优先）
        if args.dataset_name is None:
            args.dataset_name = config.get('dataset', {}).get('name', 'libero_10')
        
        if args.pretrained_path is None:
            args.pretrained_path = config.get('model', {}).get('pretrained_path')
        
        if args.lora_adapter is None:
            args.lora_adapter = config.get('model', {}).get('lora_adapter')
        
        if args.num_traj is None:
            args.num_traj = config.get('evaluation', {}).get('num_traj', 20)
        
        if args.seed is None:
            args.seed = config.get('evaluation', {}).get('seed', 20241201)
        
        if args.video_save_interval is None:
            args.video_save_interval = config.get('evaluation', {}).get('video', {}).get('save_interval', 10)
        
        if args.video_root_dir is None:
            args.video_root_dir = config.get('evaluation', {}).get('video', {}).get('root_dir', 'outs/eval_videos')
        
        if args.gpu is None:
            args.gpu = config.get('evaluation', {}).get('gpu', 0)
        
        # 自动生成metrics_path（如果未指定）
        if args.metrics_path is None:
            metrics_dir = config.get('output', {}).get('metrics_dir', 'outs/metrics')
            prefix = config.get('output', {}).get('prefix', '')
            
            # 确定checkpoint名称
            if args.lora_adapter:
                checkpoint_name = os.path.basename(args.lora_adapter)
            else:
                checkpoint_name = os.path.basename(args.pretrained_path)
            
            # 生成文件名
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}{args.dataset_name}_eval{args.num_traj}eps_{checkpoint_name}_seed{args.seed}_{timestamp}.csv"
            args.metrics_path = os.path.join(metrics_dir, filename)
    
    # 检查必填参数
    if args.pretrained_path is None:
        parser.error("必须提供 --pretrained-path 或在配置文件中指定 model.pretrained_path")
    
    if args.metrics_path is None:
        parser.error("必须提供 --metrics-path 或使用配置文件（会自动生成）")
    
    # 设置默认值（如果仍为None）
    if args.num_traj is None:
        args.num_traj = 20
    if args.dataset_name is None:
        args.dataset_name = "libero_10"
    if args.seed is None:
        args.seed = 20241201
    if args.video_save_interval is None:
        args.video_save_interval = 10
    if args.video_root_dir is None:
        args.video_root_dir = "outs/eval_videos"
    if args.gpu is None:
        args.gpu = 0
    
    return args


def _get_obs_item(obs, key, idx):
    """兼容两种 vector obs 结构"""
    if isinstance(obs, dict):
        return obs[key][idx]
    else:
        return obs[idx][key]


def set_global_seeds(seed: int):
    """统一控制随机种子"""
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


def load_language_embedding(task_name, dataset_name, policy, device):
    """加载语言嵌入"""
    lang_embed_path = os.path.join("outs/libero_embeddings", dataset_name, f"{task_name}.pt")

    if os.path.exists(lang_embed_path):
        print(f"  ✓ 加载预计算嵌入: {task_name}")
        lang_data = torch.load(lang_embed_path)
        if isinstance(lang_data, dict):
            embeddings = lang_data['embeddings']
        else:
            embeddings = lang_data
        
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        return embeddings.to(device=device, dtype=torch.bfloat16)
    else:
        print(f"  ⚠ 实时编码嵌入: {task_name}")
        instruction = extract_instruction_from_task_name(task_name)
        return policy.encode_instruction(instruction).to(device=device, dtype=torch.bfloat16)


def evaluate_single_task(
    task_id,
    policy,
    device,
    benchmark_dict,
    args,
    checkpoint_identifier
):
    """
    评估单个任务
    返回：(success_rate, num_success, task_name, instruction)
    """
    print(f"\n{'─'*80}")
    print(f"正在评估 Task {task_id}")
    print(f"{'─'*80}")
    
    task_start_time = time.time()
    
    # 获取任务信息
    task_suite = benchmark_dict[args.dataset_name]()
    task = task_suite.get_task(task_id)
    
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    
    task_name = get_task_name_from_bddl(bddl_file)
    instruction = extract_instruction_from_task_name(task_name)
    
    print(f"  任务名称: {task_name}")
    print(f"  指令: {instruction}")
    
    # 加载语言嵌入
    text_embed = load_language_embedding(task_name, args.dataset_name, policy, device)
    
    # 获取初始状态
    init_states = task_suite.get_task_init_states(task_id)
    num_init_states = init_states.shape[0]
    
    # 创建并行环境
    env_num = args.num_traj
    env_args = dict(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
    )
    
    env = SubprocVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
    )
    env.seed(args.seed)
    env.reset()
    
    # 分配初始状态
    indices = np.arange(env_num) % num_init_states
    init_states_batch = init_states[indices]
    obs = env.set_init_state(init_states_batch)
    
    # 准备图像历史和 proprio
    MAX_EPISODE_STEPS = 400
    dones = [False] * env_num
    global_steps = 0
    
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
    
    # 物理预热
    for _ in range(5):
        env.step(np.zeros((env_num, 7), dtype=np.float32))
    
    # 决定是否保存视频
    save_videos = (args.video_save_interval > 0 and task_id % args.video_save_interval == 0)
    
    if save_videos:
        video_folder = os.path.join(
            args.video_root_dir,
            f"{args.dataset_name}_task{task_id}"
        )
        os.makedirs(video_folder, exist_ok=True)
        print(f"  📹 将保存视频到: {video_folder}")
    else:
        video_folder = "outs/videos"  # 虽然不保存，但需要一个路径
    
    num_success = 0
    
    with VideoWriter(video_folder, save_video=save_videos, fps=30, single_video=True) as video_writer:
        while (global_steps < MAX_EPISODE_STEPS) and (not all(dones)):
            actions = np.zeros((env_num, 7), dtype=np.float32)
            
            # 推理
            with torch.inference_mode():
                for i in range(env_num):
                    if dones[i]:
                        continue
                    
                    # 准备图像
                    image_arrs = []
                    for t in range(2):
                        image_arrs.append(agentview_windows[i][t])
                        image_arrs.append(eye_in_hand_windows[i][t])
                        image_arrs.append(None)
                    
                    images = [
                        Image.fromarray(arr) if arr is not None else None
                        for arr in image_arrs
                    ]
                    
                    # RDT 推理
                    action_seq = policy.step(proprios[i], images, text_embed).squeeze(0)
                    action_seq = action_seq.detach().cpu().numpy()
                    
                    if np.any(np.isnan(action_seq)) or np.any(np.isinf(action_seq)):
                        continue
                    
                    actions[i] = action_seq[0]
            
            # 环境步进
            obs, reward, done, info = env.step(actions)
            global_steps += 1
            
            # 记录视频
            if save_videos:
                video_writer.append_vector_obs(
                    obs,
                    dones,
                    camera_name="agentview_image",
                    info=info
                )
            
            # 更新观测
            for i in range(env_num):
                if not dones[i]:
                    agent_img = _get_obs_item(obs, "agentview_image", i)
                    eye_img = _get_obs_item(obs, "robot0_eye_in_hand_image", i)
                    
                    agentview_windows[i].append(agent_img)
                    eye_in_hand_windows[i].append(eye_img)
                    
                    joint_states = _get_obs_item(obs, "robot0_joint_pos", i)
                    gripper_states = _get_obs_item(obs, "robot0_gripper_qpos", i)
                    proprio_np = np.concatenate([joint_states, gripper_states], axis=-1).astype(np.float32)
                    proprios[i] = torch.from_numpy(proprio_np).to(device=device, dtype=torch.bfloat16)
                    
                    if done[i]:
                        dones[i] = True
        
        # 统计成功数
        for i in range(env_num):
            num_success += int(dones[i])
    
    env.close()
    
    # 计算成功率
    success_rate = num_success / float(env_num)
    task_time = time.time() - task_start_time
    
    print(f"  ✓ 成功率: {success_rate * 100:.2f}% ({num_success}/{env_num})")
    print(f"  ⏱ 耗时: {task_time:.1f}秒")
    
    # 写入 CSV
    write_task_result_to_csv(
        args.metrics_path,
        args.dataset_name,
        task_id,
        task_name,
        instruction,
        env_num,
        num_success,
        success_rate,
        checkpoint_identifier,
        video_folder if save_videos else ""
    )
    
    return success_rate, num_success, task_name, instruction


def write_task_result_to_csv(
    metrics_path,
    dataset_name,
    task_id,
    task_name,
    instruction,
    num_traj,
    episode_done_count,
    success_rate,
    checkpoint_path,
    video_dir
):
    """将单个任务结果写入 CSV"""
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
                "num_traj",
                "episode_done_count",
                "success_rate",
                "checkpoint_path",
                "video_dir",
            ])
        
        writer.writerow([
            dataset_name,
            task_id,
            task_name,
            instruction,
            num_traj,
            episode_done_count,
            success_rate,
            checkpoint_path,
            video_dir,
        ])


def write_average_to_csv(metrics_path, dataset_name, checkpoint_path, random_seed, run_id, num_episodes):
    """计算并写入平均成功率"""
    # 读取所有任务结果
    success_rates = []
    episode_success_counts = []
    
    with open(metrics_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                success_rate = float(row['success_rate'])
                episode_done = int(row['episode_done_count'])
                success_rates.append(success_rate)
                episode_success_counts.append(episode_done)
            except (ValueError, KeyError):
                continue
    
    if len(success_rates) == 0:
        return
    
    # 计算统计指标
    total_tasks = len(success_rates)
    avg_success_rate = sum(success_rates) / total_tasks
    total_success_episodes = sum(episode_success_counts)
    total_episodes = total_tasks * num_episodes
    
    # 追加平均值
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
            checkpoint_path,
            run_id
        ])


def main():
    args = parse_args()
    
    print("="*80)
    print("RDT-LIBERO 批量评估（优化版：一次加载模型）")
    print("="*80)
    print(f"数据集: {args.dataset_name}")
    print(f"每任务评估轮次: {args.num_traj}")
    print(f"随机种子: {args.seed}")
    print(f"物理 GPU 编号: {args.gpu}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"视频保存间隔: {args.video_save_interval}")
    print("="*80)
    
    # 确定任务数量
    if args.dataset_name == "libero_10":
        task_range = range(0, 10)
    elif args.dataset_name == "libero_90":
        task_range = range(0, 90)
    elif args.dataset_name in ["libero_spatial", "libero_object", "libero_goal"]:
        task_range = range(0, 10)
    else:
        raise ValueError(f"不支持的数据集: {args.dataset_name}")
    
    # 设置随机种子
    set_global_seeds(args.seed)
    
    # 设置设备
    # 注意：因为已经设置了 CUDA_VISIBLE_DEVICES，所以这里使用 cuda:0
    # （实际的物理GPU已经通过 CUDA_VISIBLE_DEVICES 映射了）
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"\n使用设备: 物理 GPU {args.gpu} (映射为 cuda:0)")
    else:
        device = torch.device("cpu")
        print(f"\n使用设备: CPU (CUDA 不可用)")
    
    # ======== 一次性加载模型（关键优化！）========
    print("\n" + "="*80)
    print("正在加载模型（只加载一次）...")
    print("="*80)
    
    model_start_time = time.time()
    
    config_path = 'configs/base.yaml'
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    
    pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    
    policy = create_model(
        args=config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_path,
        lora_adapter_path=args.lora_adapter,
        pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
    )
    
    model_load_time = time.time() - model_start_time
    print(f"✓ 模型加载完成！耗时: {model_load_time:.1f}秒")
    
    # 确定 checkpoint 标识（用于 CSV 记录）
    checkpoint_identifier = args.lora_adapter if args.lora_adapter else args.pretrained_path
    
    # 获取 benchmark
    benchmark_dict = get_benchmark_dict()
    
    # ======== 循环评估所有任务 ========
    print("\n" + "="*80)
    print(f"开始评估 {len(task_range)} 个任务...")
    print("="*80)
    
    all_start_time = time.time()
    all_success_rates = []
    all_success_counts = []
    
    for task_id in task_range:
        success_rate, num_success, task_name, instruction = evaluate_single_task(
            task_id=task_id,
            policy=policy,
            device=device,
            benchmark_dict=benchmark_dict,
            args=args,
            checkpoint_identifier=checkpoint_identifier
        )
        
        all_success_rates.append(success_rate)
        all_success_counts.append(num_success)
    
    total_time = time.time() - all_start_time
    
    # ======== 计算并保存统计结果 ========
    print("\n" + "="*80)
    print("所有任务评估完成！正在计算统计结果...")
    print("="*80)
    
    # 生成 run_id
    import datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 写入平均值
    write_average_to_csv(
        args.metrics_path,
        args.dataset_name,
        checkpoint_identifier,
        args.seed,
        run_id,
        args.num_traj
    )
    
    # 打印汇总
    avg_success_rate = sum(all_success_rates) / len(all_success_rates)
    total_success = sum(all_success_counts)
    total_episodes = len(task_range) * args.num_traj
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 28 + "评估结果汇总" + " " * 38 + "║")
    print("╠" + "═" * 78 + "╣")
    print(f"║ 数据集           : {args.dataset_name:<58} ║")
    print(f"║ Checkpoint       : {os.path.basename(checkpoint_identifier):<58} ║")
    print(f"║ 随机种子         : {args.seed:<58} ║")
    print("╠" + "═" * 78 + "╣")
    print(f"║ 评估任务总数     : {len(task_range):<58} ║")
    print(f"║ 每任务评估轮次   : {args.num_traj:<58} ║")
    print(f"║ 总评估轮次       : {total_episodes:<58} ║")
    print("╠" + "═" * 78 + "╣")
    print(f"║ ✓ 平均成功率     : {avg_success_rate * 100:>6.2f}%{' ' * 50} ║")
    print(f"║ ✓ 成功轮次       : {total_success}/{total_episodes}{' ' * (52 - len(str(total_success)) - len(str(total_episodes)))} ║")
    print("╠" + "═" * 78 + "╣")
    print(f"║ ⏱ 模型加载时间   : {model_load_time:>6.1f}秒{' ' * 48} ║")
    print(f"║ ⏱ 评估总时间     : {total_time:>6.1f}秒{' ' * 48} ║")
    print(f"║ ⏱ 平均每任务     : {total_time/len(task_range):>6.1f}秒{' ' * 48} ║")
    print("╚" + "═" * 78 + "╝")
    
    print(f"\n✅ 评估结果已保存到: {args.metrics_path}")
    print(f"📊 总耗时: {model_load_time + total_time:.1f}秒 (模型加载 {model_load_time:.1f}秒 + 评估 {total_time:.1f}秒)")
    print("\n相比原版脚本，节省了约 {:.1f} 秒的重复模型加载时间！".format(model_load_time * (len(task_range) - 1)))


if __name__ == "__main__":
    main()

