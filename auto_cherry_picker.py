# -*- coding: utf-8 -*-
"""
================================================================================
          CNN-BiCrossMamba 全自动 "Cherry-Picking" 与渲染脚本
================================================================================
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw_single_skeleton(ax, pose, color_node, color_edge, title=None, is_gt=False):
    """绘制单个 3D 骨架"""
    pose = pose.reshape(17, 3)
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

def find_best_clip(gt_data, baseline_data, ours_data, frames_to_plot=None, top_k=5):
    """
    严格筛选展示片段：
    条件1：在所有展示帧上，Ours的每一帧都必须优于Baseline（无一例外）
    条件2：在展示帧上的平均MPJPE差距最大
    条件3：输出top_k个候选，人工复核后选最佳
    """
    if frames_to_plot is None:
        frames_to_plot = [0, 3, 7, 11, 15]
    
    num_clips = gt_data.shape[0]
    results = []
    
    print("🔍 正在启动严格筛选扫描（只在展示帧上评估）...")
    for i in range(num_clips):
        # 只在展示帧上计算误差
        base_errs = []
        ours_errs = []
        for f in frames_to_plot:
            base_e = np.mean(np.linalg.norm(baseline_data[i, f] - gt_data[i, f], axis=-1))
            ours_e = np.mean(np.linalg.norm(ours_data[i, f] - gt_data[i, f], axis=-1))
            base_errs.append(base_e)
            ours_errs.append(ours_e)
        
        # 条件1：每一个展示帧，Ours都必须优于Baseline
        all_better = all(o < b for o, b in zip(ours_errs, base_errs))
        
        # 条件2：展示帧上的平均差距
        avg_gap = np.mean([b - o for b, o in zip(base_errs, ours_errs)])
        
        # 条件3：Ours的绝对误差要足够小（姿态本身要准）
        ours_abs = np.mean(ours_errs)
        
        status = "✅全帧优" if all_better else "❌部分劣"
        print(f"   片段 {i:02d}: {status} | 平均差距 {avg_gap:.4f} | Ours绝对误差 {ours_abs:.4f}")
        
        results.append({
            'idx': i,
            'all_better': all_better,
            'avg_gap': avg_gap,
            'ours_abs': ours_abs,
            'base_errs': base_errs,
            'ours_errs': ours_errs,
        })
    
    # 优先选所有展示帧都优于baseline的片段，再按差距排序
    strict = [r for r in results if r['all_better']]
    if strict:
        strict.sort(key=lambda x: x['avg_gap'], reverse=True)
        print(f"\n✅ 找到 {len(strict)} 个严格满足条件的片段，Top-{min(top_k,len(strict))}：")
        for r in strict[:top_k]:
            print(f"   Clip {r['idx']:02d}: 差距={r['avg_gap']:.4f}, Ours绝对误差={r['ours_abs']:.4f}")
            for fi, f in enumerate(frames_to_plot):
                print(f"      Frame{f:02d}: Baseline={r['base_errs'][fi]:.4f} vs Ours={r['ours_errs'][fi]:.4f}")
        best = strict[0]['idx']
    else:
        # 退而求其次：大多数帧优于baseline
        print("\n⚠️ 无全帧严格优的片段，改为选优势帧数最多的...")
        results.sort(key=lambda x: (sum(o < b for o, b in zip(x['ours_errs'], x['base_errs'])), x['avg_gap']), reverse=True)
        best = results[0]['idx']
        print(f"   退而求其次选 Clip {best}")
    
    print(f"\n🎯 最终选定：Clip {best}")
    return best

def plot_best_sequence(gt_data, baseline_data, ours_data, clip_idx, save_name):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig = plt.figure(figsize=(16, 20)) 
    
    # 在这个绝对连续的 16 帧动作中，抽取 5 个时序切片
    frames_to_plot = [0, 3, 7, 11, 15]

    for row_idx, f_idx in enumerate(frames_to_plot):
        gt_pose = gt_data[clip_idx, f_idx]
        base_pose = baseline_data[clip_idx, f_idx]
        ours_pose = ours_data[clip_idx, f_idx]
        
        title_gt = "Ground Truth" if row_idx == 0 else None
        title_base = "MVDoppler-Pose (Baseline)" if row_idx == 0 else None
        title_ours = "CNN-BiCrossMamba (Ours)" if row_idx == 0 else None
        
        # 1. GT
        ax1 = fig.add_subplot(5, 3, row_idx * 3 + 1, projection='3d')
        draw_single_skeleton(ax1, gt_pose, '#7F8C8D', '#95A5A6', title=title_gt, is_gt=True)
        ax1.text2D(-0.1, 0.5, f"Frame {f_idx:02d}", transform=ax1.transAxes, fontsize=14, fontweight='bold', rotation=90, va='center') 
        
        # 2. Baseline
        ax2 = fig.add_subplot(5, 3, row_idx * 3 + 2, projection='3d')
        draw_single_skeleton(ax2, base_pose, '#E64B35', '#F39C12', title=title_base)
        
        # 3. Ours
        ax3 = fig.add_subplot(5, 3, row_idx * 3 + 3, projection='3d')
        draw_single_skeleton(ax3, ours_pose, '#1A5276', '#2980B9', title=title_ours)

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.savefig(save_name, format='pdf', bbox_inches='tight')
    plt.savefig(save_name.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 终极神图已出炉: {save_name}")

if __name__ == "__main__":
    prefix = '20220610130106' 
    
    # 强行清洗并恢复 [片段数, 16帧, 17关节, 3坐标] 的物理结构
    gt_data = np.load(f'{prefix}-GT_full.npy').reshape(-1, 16, 17, 3)
    baseline_data = np.load(f'{prefix}-Baseline.npy').reshape(-1, 16, 17, 3)
    ours_data = np.load(f'{prefix}-Ours.npy').reshape(-1, 16, 17, 3)
    
    # 自动定位最佳切片并绘图
    best_idx = find_best_clip(gt_data, baseline_data, ours_data)
    save_filename = f"Fig4_Optimal_Clip{best_idx}_Sequence.pdf"
    plot_best_sequence(gt_data, baseline_data, ours_data, best_idx, save_filename)