import matplotlib
matplotlib.use('Agg') # 依然使用无头模式
import matplotlib.pyplot as plt
import h5py
import numpy as np

def visualize_spectrograms(file_path):
    print(f"正在加载数据: {file_path}")
    with h5py.File(file_path, 'r') as f:
        # 提取数组
        radar_dat = np.array(f['radar_dat'])
        radar_rng = np.array(f['radar_rng'])

    print(f"成功提取 radar_dat，形状: {radar_dat.shape}")
    print(f"成功提取 radar_rng，形状: {radar_rng.shape}")

    # 创建 2x2 的画布
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # 1. 雷达一：微多普勒图 (取前 400 帧来放大看细节)
    # cmap='jet' 是雷达界最常用的热力图配色
    im1 = axs[0, 0].imshow(radar_dat[:, :400, 0], aspect='auto', cmap='jet', origin='lower')
    axs[0, 0].set_title('Radar 1 - Micro-Doppler Map')
    axs[0, 0].set_ylabel('Doppler Bins (Velocity)')
    axs[0, 0].set_xlabel('Time Frames')
    fig.colorbar(im1, ax=axs[0, 0])

    # 2. 雷达二：微多普勒图
    im2 = axs[0, 1].imshow(radar_dat[:, :400, 1], aspect='auto', cmap='jet', origin='lower')
    axs[0, 1].set_title('Radar 2 - Micro-Doppler Map')
    axs[0, 1].set_ylabel('Doppler Bins (Velocity)')
    axs[0, 1].set_xlabel('Time Frames')
    fig.colorbar(im2, ax=axs[0, 1])

    # 3. 雷达一：距离图 (Range Map)
    im3 = axs[1, 0].imshow(radar_rng[:, :, 0], aspect='auto', cmap='jet', origin='lower')
    axs[1, 0].set_title('Radar 1 - Range Map')
    axs[1, 0].set_ylabel('Range Bins (Distance)')
    axs[1, 0].set_xlabel('Time Frames')
    fig.colorbar(im3, ax=axs[1, 0])

    # 4. 雷达二：距离图 (Range Map)
    im4 = axs[1, 1].imshow(radar_rng[:, :, 1], aspect='auto', cmap='jet', origin='lower')
    axs[1, 1].set_title('Radar 2 - Range Map')
    axs[1, 1].set_ylabel('Range Bins (Distance)')
    axs[1, 1].set_xlabel('Time Frames')
    fig.colorbar(im4, ax=axs[1, 1])

    plt.tight_layout()
    save_path = 'radar_spectrograms.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 绘图完成！图像已保存为: {save_path}")
    print("👉 请在左侧文件树中下载该图片，你将看到真正的雷达图象！")

if __name__ == '__main__':
    # 填入你刚才探测出来的绝对路径
    file_path = '/root/autodl-tmp/dataset/2022Jun25-0207/20220625020830/output_3D/keypoint3D_adjusted.npz'
    visualize_spectrograms(file_path)