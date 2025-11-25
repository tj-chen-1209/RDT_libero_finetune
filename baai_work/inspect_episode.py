import os
from bson import decode_all
import numpy as np

episode_dir = "/home/zhukefei/chensiqi/RDT_libero_finetune/data/baai/data/action176/episode_0"  # TODO: 改成你的实际路径


# ========== 1) 先看 episode_0.bson 的内部结构 ==========
ep_bson_path = os.path.join(episode_dir, "episode_0.bson")
with open(ep_bson_path, "rb") as f:
    ep_docs = decode_all(f.read())

doc = ep_docs[0]
print("episode_0.bson 顶层 keys:", doc.keys())

metadata = doc["metadata"]
data = doc["data"]

print("\nmetadata keys:", metadata.keys())
print("metadata 内容预览:", metadata)

print("\ndata keys:", data.keys())
for k, v in data.items():
    if isinstance(v, list):
        print(f"  data['{k}'] 是 list，长度 {len(v)}")
        if len(v) > 0:
            print(f"    第 0 个元素类型: {type(v[0])}")
            if isinstance(v[0], dict):
                print(f"    第 0 个元素的 keys: {v[0].keys()}")
    else:
        print(f"  data['{k}'] 类型: {type(v)}")

# 如果 data 里面有 'frames' 之类的：
if "frames" in data:
    frames = data["frames"]
    print("\nepisode_0.bson 里的 frames 数量:", len(frames))
    if len(frames) > 0:
        print("frames[0] keys:", frames[0].keys())

# ========== 2) 再看 xhand_control_data.bson 里的 frames ==========
xhand_path = os.path.join(episode_dir, "xhand_control_data.bson")
with open(xhand_path, "rb") as f:
    xhand_docs = decode_all(f.read())

xdoc = xhand_docs[0]
print("\nxhand_control_data.bson 顶层 keys:", xdoc.keys())

frames = xdoc["frames"]
print("xhand_control_data.bson frames 数量:", len(frames))
if len(frames) > 0:
    print("第 0 个 frame 的 keys:", frames[0].keys())
    for k, v in frames[0].items():
        print(f"  frame['{k}'] => {type(v)}")
