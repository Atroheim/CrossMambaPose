"""
mamba_modules.py
Pure-Mamba 3D HPE 项目 —— 多模态雷达稀疏化前端

数据流：
  DataLoader [B, 4, T, R]
      │
      ├─ 拆分为 4 个单模态 [B, T, R]
      ├─ 4 个独立 ModalitySparsifier（独立权重）
      │   每个输出原始稀疏 token [B, T*top_k, 2]
      ├─ + Modality Embedding（4 种可学习向量，dim=2，加在原始特征上）
      ├─ 拼接 → [B, 4*T*top_k, 2]
      ├─ nn.Linear(2, d_model=256)
      └─ 输出 [B, seq_len, 256]  →  送入 Mamba Block

作者注：Modality Embedding 在投影之前以加法形式注入，
       物理意义：给每个 token 的 (坐标, 强度) 附加一个可学习的模态偏置，
       使投影层能够以不同方式处理距离 token 和速度 token。
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
# 模块 1：单模态稀疏器
# 职责：对单一物理量（距离图 or 速度图）做 Top-K 截断，
#       输出原始 2 维物理 token，不做投影（投影统一在外层做）
# ──────────────────────────────────────────────────────────────
class ModalitySparsifier(nn.Module):
    def __init__(self, top_k: int = 32):
        """
        :param top_k: 每帧保留的最强反射点数量
        """
        super().__init__()
        self.top_k = top_k

    def forward(self, radar_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param radar_map: 单模态雷达图，形状 [B, T, R]
                          B=batch, T=时间帧数, R=分辨率 bin 数
        :return:
            normalized_coords: 归一化坐标，形状 [B, T*top_k]，范围 [-1, 1]
            intensities:       反射强度，形状 [B, T*top_k]，量纲与输入一致
        """
        B, T, R = radar_map.shape

        # Step 1: Top-K 截断
        # 沿 R 维（距离/速度 bin）取每帧能量最强的 top_k 个点
        # values: [B, T, top_k]  —— 反射强度
        # indices: [B, T, top_k] —— 在 R 维的原始下标
        values, indices = torch.topk(radar_map, k=self.top_k, dim=-1)

        # Step 2: 坐标归一化到 [-1.0, 1.0]
        # 目的：消除 R 取值范围差异对数值稳定性的影响
        # 公式：coord = (index / R) * 2 - 1
        normalized_coords = (indices.float() / R) * 2.0 - 1.0  # [B, T, top_k]

        # Step 3: 展平时间和点数维度，合并为一维序列
        # [B, T, top_k] → [B, T*top_k]
        normalized_coords = normalized_coords.reshape(B, T * self.top_k)
        intensities = values.reshape(B, T * self.top_k)

        return normalized_coords, intensities


# ──────────────────────────────────────────────────────────────
# 模块 2：多模态雷达前端（主模块）
# 职责：
#   1. 将 [B, 4, T, R] 输入按模态拆分，送入 4 个独立 Sparsifier
#   2. 对每个模态的 token 注入可学习 Modality Embedding
#   3. 拼接四模态序列，线性投影到 d_model
# ──────────────────────────────────────────────────────────────
class MultiModalRadarFrontend(nn.Module):

    # 固定模态顺序，与 DataLoader 的通道顺序严格对应
    MODALITY_NAMES = ["V1_Range", "V1_Doppler", "V2_Range", "V2_Doppler"]

    def __init__(
        self,
        top_k: int = 32,
        d_model: int = 256,
        num_modalities: int = 4,
    ):
        """
        :param top_k:          每帧每模态保留的最强反射点数
        :param d_model:        输出特征维度，需与后续 Mamba block 的 d_model 严格一致
        :param num_modalities: 模态数，固定为 4（V1_Range/V1_Doppler/V2_Range/V2_Doppler）
        """
        super().__init__()
        self.top_k = top_k
        self.d_model = d_model
        self.num_modalities = num_modalities

        # 4 个独立 Sparsifier，各自有独立参数（此处无可学习参数，
        # 但保持独立实例是架构规范，便于后续扩展，如独立的 BN/归一化层）
        self.sparsifiers = nn.ModuleList(
            [ModalitySparsifier(top_k=top_k) for _ in range(num_modalities)]
        )

        # Modality Embedding：4 种模态各一个可学习的 2 维向量
        # 物理意义：在 (归一化坐标, 反射强度) 的原始物理空间中，
        #           为每种模态注入一个可学习偏置，
        #           使后续线性投影层能以不同方式处理距离 token 和速度 token
        # 形状：[num_modalities, 2]，即每个模态一个 2 维向量
        self.modality_embeddings = nn.Embedding(num_modalities, 2)

        # 线性投影：将 2 维物理 token 投影到 Mamba 所需的高维空间
        # 输入 dim=2：(归一化坐标 + 模态偏置_0, 反射强度 + 模态偏置_1)
        # 输出 dim=d_model：Mamba block 的标准输入维度
        self.input_proj = nn.Linear(2, d_model)

    def forward(self, radar_input: torch.Tensor) -> torch.Tensor:
        """
        :param radar_input: DataLoader 输出的原始多模态雷达张量
                            形状 [B, 4, T, R]
                            通道顺序：0=V1_Range, 1=V1_Doppler, 2=V2_Range, 3=V2_Doppler
        :return: mamba_ready: 形状 [B, 4*T*top_k, d_model]，可直接送入 Mamba block
        """
        B, num_mod, T, R = radar_input.shape
        assert num_mod == self.num_modalities, \
            f"输入通道数 {num_mod} 与初始化的 num_modalities={self.num_modalities} 不匹配"

        token_list = []  # 收集 4 个模态的 token，最后拼接

        for mod_idx in range(self.num_modalities):

            # Step 1: 取出单模态雷达图
            # [B, 4, T, R] → [B, T, R]
            single_map = radar_input[:, mod_idx, :, :]

            # Step 2: 独立 Sparsifier 提取稀疏点
            # coords: [B, T*top_k]  intensities: [B, T*top_k]
            coords, intensities = self.sparsifiers[mod_idx](single_map)

            # Step 3: 打包为 2 维物理 token
            # stack 后：[B, T*top_k, 2]，dim=-1 对应 (坐标, 强度)
            tokens = torch.stack([coords, intensities], dim=-1)  # [B, T*top_k, 2]

            # Step 4: 注入 Modality Embedding
            # modality_embeddings(mod_idx_tensor) → [2]
            # 广播加到 [B, T*top_k, 2] 上，为每个 token 附加模态偏置
            mod_idx_tensor = torch.tensor(mod_idx, device=radar_input.device)
            mod_emb = self.modality_embeddings(mod_idx_tensor)  # [2]
            tokens = tokens + mod_emb  # 广播：[B, T*top_k, 2] + [2]

            token_list.append(tokens)  # 存入列表

        # Step 5: 拼接 4 个模态的 token 序列（沿序列维度拼接）
        # 4 × [B, T*top_k, 2] → [B, 4*T*top_k, 2]
        all_tokens = torch.cat(token_list, dim=1)

        # Step 6: 线性投影到 d_model
        # [B, 4*T*top_k, 2] → [B, 4*T*top_k, d_model]
        mamba_ready = self.input_proj(all_tokens)

        return mamba_ready


# ──────────────────────────────────────────────────────────────
# 单元测试
# 运行方式：conda activate mambapose && python mamba_modules.py
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 超参数（与正式训练保持一致）
    BATCH     = 2
    T         = 16    # 时间帧数
    R         = 256   # 分辨率 bin 数
    TOP_K     = 32    # 每帧每模态保留点数
    D_MODEL   = 256   # Mamba d_model

    # 模拟 DataLoader 输出：[B, 4, T, R]，4 个模态已按顺序排好
    dummy_input = torch.rand(BATCH, 4, T, R)

    # 实例化前端模块
    frontend = MultiModalRadarFrontend(top_k=TOP_K, d_model=D_MODEL)

    # 前向传播
    output = frontend(dummy_input)

    # 打印形状验证
    print("=" * 50)
    print(f"输入雷达张量形状:   {dummy_input.shape}")
    # 期望: torch.Size([2, 4, 16, 256])

    print(f"输出 Mamba 序列形状: {output.shape}")
    # 期望: torch.Size([2, 2048, 256])
    # seq_len = 4 modalities × 16 frames × 32 points = 2048

    expected_seq_len = 4 * T * TOP_K
    assert output.shape == (BATCH, expected_seq_len, D_MODEL), "形状验证失败！"
    print(f"形状验证通过 ✓  seq_len = 4×{T}×{TOP_K} = {expected_seq_len}")
    print("=" * 50)