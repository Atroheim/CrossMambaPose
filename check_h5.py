"""
probe_radar_data.py
在服务器上运行，探查 .h5 原始雷达数据的完整物理结构。

运行方式：
  conda activate mambapose
  python probe_radar_data.py
"""

import os
import h5py
import numpy as np

# ── 替换为任意一个真实的 .h5 文件路径
H5_PATH = '/root/autodl-tmp/dataset/2022Jul13-1744/radar_v2/20220713174516.h5'   # 先找一个样本

def find_first_h5(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith('.h5'):
                return os.path.join(dirpath, f)
    return None

h5_path = find_first_h5(H5_PATH)
print(f"找到文件: {h5_path}\n")

hf = h5py.File(h5_path, 'r')

print("=" * 60)
print("【所有 key】")
def print_keys(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  Dataset: {name:30s}  shape={obj.shape}  dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"  Group:   {name}")
hf.visititems(print_keys)

print("\n【attrs（文件级元数据）】")
for k, v in hf.attrs.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 60)

# 重点打印 radar_dat 和 radar_rng 的详细统计
for key in ['radar_dat', 'radar_rng']:
    if key in hf:
        arr = np.array(hf[key])
        print(f"\n【{key}】")
        print(f"  shape : {arr.shape}")
        print(f"  dtype : {arr.dtype}")
        print(f"  min   : {arr.min():.6f}")
        print(f"  max   : {arr.max():.6f}")
        print(f"  mean  : {arr.mean():.6f}")
        print(f"  物理维度解读（猜测）:")
        for i, s in enumerate(arr.shape):
            print(f"    dim{i}: size={s}")

# 检查是否有其他物理量 key
other_keys = [k for k in hf.keys() if k not in ('radar_dat', 'radar_rng')]
if other_keys:
    print(f"\n【其他 key】: {other_keys}")
    for k in other_keys:
        arr = np.array(hf[k])
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")

hf.close()

# 同时查一个 keypoint 文件
print("\n" + "=" * 60)
print("【查找对应 keypoints.npy】")
kp_dir = os.path.dirname(h5_path).replace('radar_v2', '')
# 尝试常见路径
for candidate in [
    os.path.join(kp_dir, 'output_3D', 'keypoints.npy'),
    os.path.join(kp_dir, 'output_3D', 'keypoints.npz'),
]:
    if os.path.exists(candidate):
        kp = np.load(candidate, allow_pickle=True)
        if hasattr(kp, 'shape'):
            print(f"  {candidate}: shape={kp.shape}, dtype={kp.dtype}")
        else:
            print(f"  {candidate}: keys={list(kp.keys())}")
        break
else:
    print("  未找到，请手动确认路径")