#!/usr/bin/env python3
"""
测试 extract_instruction 函数
使用 libero_10 数据集中的实际文件进行测试
"""

import os
import glob


def extract_instruction(file_path):
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
    # 获取文件名（不含路径和扩展名）
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


def main():
    # 获取 libero_10 数据集中的所有文件
    dataset_dir = "data/datasets/libero_10"
    hdf5_files = glob.glob(os.path.join(dataset_dir, "*.hdf5"))
    
    print("=" * 80)
    print("测试 extract_instruction 函数")
    print("=" * 80)
    print(f"\n找到 {len(hdf5_files)} 个 HDF5 文件\n")
    
    # 测试所有文件
    for i, file_path in enumerate(sorted(hdf5_files), 1):
        filename = os.path.basename(file_path)
        instruction = extract_instruction(file_path)
        
        print(f"[{i}/{len(hdf5_files)}]")
        print(f"文件名: {filename}")
        print(f"提取的指令: {instruction}")
        print("-" * 80)
    
    # 测试一些边界情况
    print("\n" + "=" * 80)
    print("测试边界情况")
    print("=" * 80)
    
    test_cases = [
        "KITCHEN_SCENE3_turn_on_the_stove_demo.hdf5",
        "KITCHEN_SCENE10_some_task_demo.hdf5",  # SCENE10 特殊情况
        "STUDY_SCENE1_pick_up_the_book_demo.hdf5",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_demo.hdf5",
        "pick_up_the_black_bowl_demo.hdf5",  # 不以大写开头
    ]
    
    for test_file in test_cases:
        instruction = extract_instruction(test_file)
        print(f"\n文件名: {test_file}")
        print(f"提取的指令: {instruction}")


if __name__ == "__main__":
    main()






