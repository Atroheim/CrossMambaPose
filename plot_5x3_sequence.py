# -*- coding: utf-8 -*-
"""
================================================================================
          CNN-BiCrossMamba 5x3 连续动作定性对比图 (Publication-Ready)
================================================================================
功能：抽取 5 个等间距时间帧，生成 5行 x 3列 的终极动作序列对比图。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw_single_skeleton(ax, pose, color_node, color_edge, title=None, is_gt=False):
    """绘制单个 3D 骨架 (防弹清洗版)"""
    pose = np.asarray(pose, dtype=float).reshape(17, 3)
    skeleton_bones = [
        [0, 1], [1, 3], [0, 2], [2, 4],     
        [5, 7], [7, 9], [6, 8], [8, 10],    
        [5, 6], [5, 11], [6, 12], [11, 12], 
        [11, 13], [13, 15], [12, 14], [14, 16] 
    ]
    
    marker_style = 'o' if is_gt else '^'
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], color=color_node, marker=marker_style, s=30, alpha=0.8)
    
    line_style = '--' if is_gt else '-'
    line_width = 2.0 if is_gt else 3.0
    for bone in skeleton_bones:
        p1, p2 = bone
        xs, ys, zs = [float(pose[p1, 0]), float(pose[p2, 0])], [float(pose[p1, 1]), float(pose[p2, 1])], [float(pose[p1, 2]), float(pose[p2, 2])]
        ax.plot(xs, ys, zs, color=color_edge, linestyle=line_style, linewidth=line_width, alpha=0.8)
    
    # 锁定绝对坐标系，确保 15 个子图不会因为动作幅度而在视觉上缩放变形
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.view_init(elev=15, azim=60)

def plot_5x3_sequence(gt_data, baseline_data, ours_data, frames=[0, 3, 7, 11, 15], save_name="Fig4_5x3_Sequence.pdf"):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # 画布被进一步拉长，高度设为 20 以容纳 5 行
    fig = plt.figure(figsize=(16, 20)) 
    
    # 自适应提取护盾
    def extract_frame(data, f_idx):
        # 不管传进来的是 (16, 17, 3) 还是 (10, 16, 17, 3)
        # 直接把前面所有的维度强行揉碎，变成一个纯净的帧列表 (N, 17, 3)
        arr = np.asarray(data, dtype=float).reshape(-1, 17, 3)
        
        # 安全锁：防止你填的帧数超出了视频总长度
        safe_idx = min(f_idx, arr.shape[0] - 1)
        return arr[safe_idx]

    for row_idx, f_idx in enumerate(frames):
        gt_pose = extract_frame(gt_data, f_idx)
        base_pose = extract_frame(baseline_data, f_idx)
        ours_pose = extract_frame(ours_data, f_idx)
        
        # 仅在第一行显示大标题
        title_gt = "Ground Truth" if row_idx == 0 else None
        title_base = "MVDoppler-Pose (Baseline)" if row_idx == 0 else None
        title_ours = "CNN-BiCrossMamba (Ours)" if row_idx == 0 else None
        
        # 绘制 GT (列1)
        ax1 = fig.add_subplot(5, 3, row_idx * 3 + 1, projection='3d')
        draw_single_skeleton(ax1, gt_pose, '#7F8C8D', '#95A5A6', title=title_gt, is_gt=True)
        # 添加极具学术感的纵向时间轴标签
        ax1.text2D(-0.1, 0.5, f"Frame {f_idx:02d}", transform=ax1.transAxes, fontsize=14, fontweight='bold', rotation=90, va='center') 
        
        # 绘制 Baseline (列2)
        ax2 = fig.add_subplot(5, 3, row_idx * 3 + 2, projection='3d')
        draw_single_skeleton(ax2, base_pose, '#E64B35', '#F39C12', title=title_base)
        
        # 绘制 Ours (列3)
        ax3 = fig.add_subplot(5, 3, row_idx * 3 + 3, projection='3d')
        draw_single_skeleton(ax3, ours_pose, '#1A5276', '#2980B9', title=title_ours)

    plt.subplots_adjust(wspace=0.05, hspace=0.1) # 压缩间距，让图片更紧凑
    plt.savefig(save_name, format='pdf', bbox_inches='tight')
    plt.savefig(save_name.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 史诗级 5x3 连续序列图已生成: {save_name}")

if __name__ == "__main__":
    prefix = '20220610130106' 
    gt_data = np.load(f'{prefix}-GT_full.npy')
    baseline_data = np.load(f'{prefix}-Baseline.npy')
    ours_data = np.load(f'{prefix}-Ours.npy') 
    

    plot_5x3_sequence(gt_data, baseline_data, ours_data, frames=[0, 30, 60, 90, 120])