# 必须在最前面设置 Agg 模式
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py  # <--- 引入专门读取 HDF5 格式的库

def visualize_radar_and_skeleton(frame_idx=100):
    # 1. 加载数据
    print("Loading data...")
    kpts_path = '/root/autodl-tmp/dataset/2022Jun25-0207/20220625020830/output_3D/keypoints.npy'
    radar_path = '/root/autodl-tmp/dataset/2022Jun25-0207/20220625020830/output_3D/keypoint3D_adjusted.npz'
    
    # keypoints 确实是正常的 numpy 文件
    kpts_data = np.load(kpts_path, allow_pickle=True)          
    
    # 【核心修改】伪装成 .npz 的 HDF5 文件必须用 h5py 读取
    radar_data = h5py.File(radar_path, 'r')

    # 从 HDF5 中提取数据并转换为一维的 numpy 数组
    frames = np.array(radar_data['frame']).flatten()
    x = np.array(radar_data['x']).flatten()
    y = np.array(radar_data['y']).flatten()
    
    # 如果雷达数据中含有 Z 坐标则读取，如果没有，则默认置为 0
    if 'z' in radar_data.keys():
        z = np.array(radar_data['z']).flatten()
    else:
        z = np.zeros_like(x)

    # 2. 提取指定帧的雷达点
    mask = (frames == frame_idx)
    rx, ry, rz = x[mask], y[mask], z[mask]

    print(f"Frame {frame_idx} 提取到了 {len(rx)} 个雷达点。")

    # 3. 提取指定帧的人体关键点
    if frame_idx >= len(kpts_data):
        print(f"Error: 帧数 {frame_idx} 超出范围 (最大 {len(kpts_data)-1})")
        # 记得关闭文件
        radar_data.close()
        return
    kpts = kpts_data[frame_idx]

    # 4. 开始绘图
    print(f"Generating plot for frame {frame_idx}...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 4.1 绘制雷达点云 (红色散点)
    ax.scatter(rx, ry, rz, c='r', marker='o', s=30, label='Radar Point Cloud')

    # 4.2 绘制人体关键点 (蓝色三角)
    ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2], c='b', marker='^', s=50, label='GT Keypoints')

    # 4.3 绘制骨架连线 (COCO 17关键点标准连线)
    skeleton_bones = [
        [0, 1], [1, 3], [0, 2], [2, 4],  # 头部
        [5, 7], [7, 9], [6, 8], [8, 10], # 手臂
        [5, 6], [5, 11], [6, 12], [11, 12], # 躯干
        [11, 13], [13, 15], [12, 14], [14, 16] # 腿部
    ]
    for bone in skeleton_bones:
        p1, p2 = bone
        ax.plot([kpts[p1, 0], kpts[p2, 0]], 
                [kpts[p1, 1], kpts[p2, 1]], 
                [kpts[p1, 2], kpts[p2, 2]], 'b-', alpha=0.6)

    # 设置坐标轴标签和图例
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(f'MVDoppler Frame {frame_idx}: Radar Points vs 3D Skeleton')
    ax.legend()
    
    # 调整视角以便于观察
    ax.view_init(elev=20, azim=45)
    
    # 5. 保存图片到本地
    save_path = f'visualization_frame{frame_idx}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 释放 HDF5 文件内存
    radar_data.close()
    
    print(f"✅ 绘图完成！图片已保存为: {save_path}")

if __name__ == "__main__":
    visualize_radar_and_skeleton(frame_idx=100)