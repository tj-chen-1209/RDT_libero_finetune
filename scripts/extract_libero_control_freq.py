import os
import json
import h5py
import numpy as np
from pathlib import Path


def _to_str(x):
    """稳一点的 attr -> str 转换"""
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.ndarray):
        x = x[()]  # 标量
    if isinstance(x, (np.bytes_, np.str_)):
        return str(x)
    return x


def extract_control_freq_from_hdf5(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            if 'data' not in f:
                print(f"警告: {file_path} 中没有 'data' 组")
                return None

            data_group = f['data']
            attrs = data_group.attrs

            # env_name / env
            env_name = None
            for k in ['env_name', 'env']:
                if k in attrs:
                    env_name = _to_str(attrs[k])
                    break

            env_info = None
            control_freq = None

            # env_args / env_info 兜底
            env_args_key = None
            for k in ['env_args', 'env_info']:
                if k in attrs:
                    env_args_key = k
                    break

            if env_args_key is not None:
                env_args_str = _to_str(attrs[env_args_key])

                try:
                    env_info = json.loads(env_args_str)
                    if isinstance(env_info, dict):
                        env_kwargs = env_info.get('env_kwargs', {})
                        if isinstance(env_kwargs, dict):
                            control_freq = env_kwargs.get('control_freq')

                            # 再兜底 controller_configs 里
                            if control_freq is None:
                                cc = env_kwargs.get('controller_configs', {})
                                if isinstance(cc, dict):
                                    control_freq = cc.get('control_freq')

                        # 顶层兜底
                        if control_freq is None:
                            control_freq = env_info.get('control_freq')

                except json.JSONDecodeError as e:
                    print(f"警告: 解析 {env_args_key} JSON 失败 ({file_path}): {e}")
                    env_info = env_args_str  # 原始字符串保存一下

            return {
                'file_path': file_path,
                'env_name': env_name,
                'env_info': env_info,
                'control_freq': control_freq,
            }

    except Exception as e:
        print(f"错误: 处理文件 {file_path} 时出错: {e}")
        return None


def process_libero_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        print(f"错误: 目录不存在: {dataset_dir}")
        return []

    # 如果有子目录建议用 rglob
    hdf5_files = list(dataset_dir.rglob('*.hdf5'))
    if not hdf5_files:
        print(f"警告: 在 {dataset_dir} 中没有找到 hdf5 文件")
        return []

    print(f"找到 {len(hdf5_files)} 个 hdf5 文件")

    results = []
    for file_path in sorted(hdf5_files):
        print(f"处理: {file_path}")
        result = extract_control_freq_from_hdf5(str(file_path))
        if result:
            results.append(result)

    return results


def main():
    # libero_10 数据集路径
    libero_10_dir = "data/datasets/libero_90"

    print("=" * 80)
    print("提取 libero_10 数据集的控制频率")
    print("=" * 80)

    results = process_libero_dataset(libero_10_dir)

    if not results:
        print("没有成功处理任何文件")
        return

    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)

    # 统计控制频率
    control_freqs = {}
    for result in results:
        freq = result['control_freq']
        if freq is not None:
            control_freqs[freq] = control_freqs.get(freq, 0) + 1

    print(f"\n控制频率统计:")
    for freq, count in sorted(control_freqs.items()):
        print(f"  {freq} Hz: {count} 个文件")

    # 显示每个文件的详细信息
    print(f"\n详细结果 ({len(results)} 个文件):")
    print("-" * 80)
    for result in results:
        print(f"\n文件: {Path(result['file_path']).name}")
        print(f"  env_name: {result['env_name']}")
        print(f"  control_freq: {result['control_freq']} Hz")
        if isinstance(result['env_info'], dict):
            env_kwargs = result['env_info'].get('env_kwargs', {})
            if env_kwargs:
                print(f"  env_kwargs 中的关键信息:")
                for key in ['control_freq', 'camera_heights', 'camera_widths']:
                    if key in env_kwargs:
                        print(f"    {key}: {env_kwargs[key]}")

    # 如果所有文件有相同的控制频率，输出建议的配置
    if len(control_freqs) == 1:
        freq = list(control_freqs.keys())[0]
        print(f"\n" + "=" * 80)
        print(f"建议: 所有文件使用相同的控制频率 {freq} Hz")
        print(f"可以在 configs/dataset_control_freq.json 中添加:")
        print(f'  "libero_10": {freq}')
        print("=" * 80)
    elif len(control_freqs) > 1:
        print(f"\n" + "=" * 80)
        print(f"警告: 发现多个不同的控制频率")
        print(f"建议: 使用最常见的频率，或根据实际情况选择")
        print("=" * 80)


if __name__ == "__main__":
    main()
