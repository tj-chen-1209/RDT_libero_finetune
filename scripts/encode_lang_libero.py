import os
import sys
import torch
import yaml
import glob
from tqdm import tqdm

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curr_dir, ".."))
from models.multimodal_encoder.t5_encoder import T5Embedder


GPU = 0
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
SAVE_DIR = "outs/libero_embeddings/"

# Note: if your GPU VRAM is less than 24GB, 
# it is recommended to enable offloading by specifying an offload directory.
OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.

def extract_instruction_from_file_path(file_path):
    """
    从 HDF5 文件名中提取任务指令
    
    Args:
        file_path: HDF5 文件路径，例如：
            "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
            "pick_up_the_black_bowl_and_place_it_on_the_plate_demo.hdf5"
    
    Returns:
        str: 提取的指令，例如：
            "turn on the stove and put the moka pot on it"
            "pick up the black bowl and place it on the plate"
    """
    task_name = os.path.basename(file_path).replace('_demo.hdf5', '')
        # 如果文件名以大写字母开头（如 KITCHEN_SCENE3_...）
    if task_name and task_name[0].isupper():
        # 查找 SCENE 的位置
        scene_pos = task_name.find("SCENE")
        if scene_pos != -1:
            # 检查是否是 SCENE10（需要跳过 8 个字符：SCENE10_）
            if "SCENE10" in task_name:
                # 从 SCENE10_ 之后开始提取
                language_part = task_name[scene_pos + 8:]
            else:
                # 从 SCENE#_ 之后开始提取（跳过 7 个字符：SCENE + 数字 + _）
                # 例如：SCENE3_ -> 跳过 7 个字符
                language_part = task_name[scene_pos + 7:]
            
            # 将下划线替换为空格
            instruction = language_part.replace('_', ' ')
        else:
            # 没有找到 SCENE，直接替换所有下划线
            instruction = task_name.replace('_', ' ')
    else:
        # 文件名不以大写字母开头，直接替换所有下划线
        instruction = task_name.replace('_', ' ')

    return instruction.strip()

def encode_libero_tasks(dataset_name="libero_10"):
    """
    为 LIBERO 数据集的所有任务生成语言嵌入
    
    Args:
        dataset_name: "libero_10" or "libero_90"
    """
    # 加载配置
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    # 初始化文本编码器
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    
    # 获取数据集中的所有 HDF5 文件
    dataset_dir = f"data/datasets/{dataset_name}"
    hdf5_files = glob.glob(os.path.join(dataset_dir, "*.hdf5"))
    
    print(f"Found {len(hdf5_files)} tasks in {dataset_name}")
    
    # 创建保存目录
    save_dir = os.path.join(SAVE_DIR, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取所有唯一的指令
    task_instructions = {}
    for file_path in hdf5_files:
        instruction = extract_instruction_from_file_path(file_path)
        task_name = os.path.basename(file_path).replace('_demo.hdf5', '')
        task_instructions[task_name] = instruction
    
    # 批量编码
    print(f"\nEncoding {len(task_instructions)} unique instructions...")
    
    for task_name, instruction in tqdm(task_instructions.items()):
        # Tokenize
        tokenized_res = tokenizer(
            instruction, 
            return_tensors="pt",
            padding="longest",
            truncation=True
        )
        tokens = tokenized_res["input_ids"].to(device)
        attn_mask = tokenized_res["attention_mask"].to(device)
        
        # Encode
        with torch.no_grad():
            text_embeds = text_encoder(
                input_ids=tokens,
                attention_mask=attn_mask
            )["last_hidden_state"].detach().cpu()
        
        # 3) 去掉 batch 维 + 去掉 padding，只保留真实 token
        text_embeds = text_embeds.squeeze(0)        # [L_max, D]
        attn_mask = attn_mask.cpu().bool().squeeze(0)  # [L_max]
        text_embed = text_embeds[attn_mask]         # [L_i, D]  只要有效 token

        # 4) 直接把 [L_i, D] 这个 Tensor 存起来
        save_path = os.path.join(save_dir, f"{task_name}.pt")
        torch.save(text_embed, save_path)
    
    print(f"\nAll embeddings saved to: {save_dir}")
    
    # 创建一个映射文件，方便查看
    mapping_file = os.path.join(save_dir, "task_instruction_mapping.txt")
    with open(mapping_file, "w") as f:
        for task_name, instruction in sorted(task_instructions.items()):
            f.write(f"{task_name}\n")
            f.write(f"  -> {instruction}\n\n")
    
    print(f"Task-instruction mapping saved to: {mapping_file}")


if __name__ == "__main__":
    # 为 LIBERO-10 生成语言嵌入
    print("=" * 80)
    print("Encoding LIBERO-10 tasks...")
    print("=" * 80)
    encode_libero_tasks("libero_10")
    
    # 为 LIBERO-90 生成语言嵌入
    print("\n" + "=" * 80)
    print("Encoding LIBERO-90 tasks...")
    print("=" * 80)
    encode_libero_tasks("libero_90")
