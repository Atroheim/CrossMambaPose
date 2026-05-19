# -*- coding: utf-8 -*-
"""
================================================================================
          CNN-BiCrossMamba 1x3 并排定性对比图 (Publication-Ready)
================================================================================
功能：在同一张画布上，使用绝对统一的视角和比例尺，并排渲染 GT、Baseline 和 Ours。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw_single_skeleton(ax, pose, color_node, color_edge, title, is_gt=False):
    """在指定的子图 (ax) 上绘制单个 3D 骨架"""
    
    # ==========================================
    # 💡 终极防线：强制清洗数据格式与维度
    # 将输入强行转为 Python 标准浮点数矩阵，彻底消除 Matplotlib 3D 绘图 Bug
    # ==========================================
    pose = np.asarray(pose, dtype=float).reshape(17, 3)

    # COCO 17 骨架连线拓扑
    skeleton_bones = [
        [0, 1], [1, 3], [0, 2], [2, 4],     
        [5, 7], [7, 9], [6, 8], [8, 10],    
        [5, 6], [5, 11], [6, 12], [11, 12], 
        [11, 13], [13, 15], [12, 14], [14, 16] 
    ]
    
    # 画点
    marker_style = 'o' if is_gt else '^'
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], 
               color=color_node, marker=marker_style, s=40, alpha=0.8)
    
    # 画线
    line_style = '--' if is_gt else '-'
    line_width = 2.0 if is_gt else 3.0
    for bone in skeleton_bones:
        p1, p2 = bone
        # 💡 抽离成纯粹的 Python 列表，避免被底层框架误识别为张量
        xs = [float(pose[p1, 0]), float(pose[p2, 0])]
        ys = [float(pose[p1, 1]), float(pose[p2, 1])]
        zs = [float(pose[p1, 2]), float(pose[p2, 2])]
        ax.plot(xs, ys, zs, color=color_edge, linestyle=line_style, linewidth=line_width, alpha=0.8)
    
    # 统一坐标轴标签和范围
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    
    ax.set_xlabel('X (m)', fontweight='bold')
    ax.set_ylabel('Y (m)', fontweight='bold')
    ax.set_zlabel('Z (m)', fontweight='bold')
    
    # 去除难看的灰色背景板
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.view_init(elev=15, azim=60) # 统一视角

def plot_1x3_comparison(gt_pose, baseline_pose, ours_pose, frame_idx=0, save_name="Fig4_1x3_Comparison.pdf"):
    """主渲染函数：生成 1x3 并排对比图"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # ==========================================
    # 💡 终极护盾：自适应降维切片
    # 不管外面传进来的是单帧(17,3)还是整个视频(16,17,3)，这里统统自动提取好！
    # ==========================================
    def extract_frame(pose_data, f_idx):
        arr = np.asarray(pose_data, dtype=float).squeeze()
        if arr.ndim == 3: # 发现是完整的 16 帧数据 (16, 17, 3)
            safe_idx = min(f_idx, arr.shape[0] - 1) # 防止选的帧数越界
            return arr[safe_idx]
        return arr # 如果已经是单帧 (17, 3)，直接返回
    
    # 强行清洗这三颗“龙珠”
    gt_pose = extract_frame(gt_pose, frame_idx)
    baseline_pose = extract_frame(baseline_pose, frame_idx)
    ours_pose = extract_frame(ours_pose, frame_idx)
    
    # 创建 1x3 的大画布
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Ground Truth (左)
    ax1 = fig.add_subplot(131, projection='3d')
    draw_single_skeleton(ax1, gt_pose, color_node='#7F8C8D', color_edge='#95A5A6', title="Ground Truth", is_gt=True)
    
    # 2. Baseline (中)
    ax2 = fig.add_subplot(132, projection='3d')
    draw_single_skeleton(ax2, baseline_pose, color_node='#E64B35', color_edge='#F39C12', title="MVDoppler-Pose (Baseline)")
    
    # 3. Ours (右)
    ax3 = fig.add_subplot(133, projection='3d')
    draw_single_skeleton(ax3, ours_pose, color_node='#1A5276', color_edge='#2980B9', title="CNN-BiCrossMamba (Ours)")
    
    plt.tight_layout()
    plt.savefig(save_name, format='pdf', bbox_inches='tight')
    plt.savefig(save_name.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 1x3 高清学术对比图已生成: {save_name}")

# ==========================================
# 模拟运行测试 (你只需把这里替换成你真实的 Numpy 数组)
# ==========================================
if __name__ == "__main__":

    prefix = '20220610130106'  # 👈 确保这里是你截图里的这串数字！
    
    print(f"正在加载动作 {prefix} 的骨架数据...")
    gt_data = np.load(f'{prefix}-GT_full.npy')
    baseline_data = np.load(f'{prefix}-Baseline.npy')
    ours_data = np.load(f'{prefix}-Ours.npy') 
    

    frame_idx = 5  # 👈 你可以通过改这个数字（比如改成 8, 12, 15）来挑选最帅的姿势
    
    gt_pose = gt_data[frame_idx]
    baseline_pose = baseline_data[frame_idx]
    ours_pose = ours_data[frame_idx]
    

    save_filename = f"Fig4_Comparison_{prefix}_Frame{frame_idx}.pdf"
    plot_1x3_comparison(gt_pose, baseline_pose, ours_pose, frame_idx=frame_idx, save_name=save_filename)