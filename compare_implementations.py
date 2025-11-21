#!/usr/bin/env python3
"""
比较自定义实现和 LIBERO 官方实现的结果
"""

import os
import sys

# 添加 LIBERO 路径（如果需要）
# sys.path.insert(0, '/path/to/libero')

try:
    from libero.libero.benchmark import grab_language_from_filename
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    print("警告: 无法导入 LIBERO 的 grab_language_from_filename 函数")


def extract_instruction(file_path):
    """自定义实现"""
    task_name = os.path.basename(file_path).replace('_demo.hdf5', '')
    if task_name and task_name[0].isupper():
        scene_pos = task_name.find("SCENE")
        if scene_pos != -1:
            if "SCENE10" in task_name:
                language_part = task_name[scene_pos + 8:]
            else:
                language_part = task_name[scene_pos + 7:]
            instruction = language_part.replace('_', ' ')
        else:
            instruction = task_name.replace('_', ' ')
    else:
        instruction = task_name.replace('_', ' ')
    return instruction.strip()


def main():
    # 测试文件列表
    test_files = [
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5",
        "KITCHEN_SCENE10_some_task_demo.hdf5",  # SCENE10 测试
    ]
    
    print("=" * 100)
    print("比较自定义实现和 LIBERO 官方实现")
    print("=" * 100)
    
    all_match = True
    
    for test_file in test_files:
        # 自定义实现
        custom_result = extract_instruction(test_file)
        
        # LIBERO 官方实现（如果可用）
        if LIBERO_AVAILABLE:
            # LIBERO 函数期望输入是 task_name + ".bddl"
            task_name = test_file.replace('_demo.hdf5', '')
            libero_result = grab_language_from_filename(task_name + ".bddl")
            
            match = custom_result == libero_result
            all_match = all_match and match
            
            status = "✓" if match else "✗"
            print(f"\n{status} 文件名: {test_file}")
            print(f"  自定义结果: {custom_result}")
            print(f"  LIBERO结果: {libero_result}")
            if not match:
                print(f"  ⚠️  结果不一致！")
        else:
            print(f"\n文件名: {test_file}")
            print(f"提取结果: {custom_result}")
    
    if LIBERO_AVAILABLE:
        print("\n" + "=" * 100)
        if all_match:
            print("✓ 所有测试用例结果一致！")
        else:
            print("✗ 部分测试用例结果不一致，请检查实现")
        print("=" * 100)


if __name__ == "__main__":
    main()






