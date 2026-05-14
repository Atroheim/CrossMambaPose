"""
mamba_encoder.py
Pure-Mamba 3D HPE 项目 —— 完整编码器前向通路

结构：
  MultiModalRadarFrontend  →  [B, 2048, 256]
        │
  MambaEncoder (N层 Mamba block堆叠)
        │
  输出特征  →  [B, 2048, 256]  （后续接姿态回归头）

运行方式：
  conda activate mambapose && python mamba_encoder.py
"""

import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

# ──────────────────────────────────────────────────────────────
# 从 mamba_modules.py 复用的前端模块（直接粘贴，避免跨文件依赖）
# ──────────────────────────────────────────────────────────────
class ModalitySparsifier(nn.Module):
    def __init__(self, top_k: int = 32):
        super().__init__()
        self.top_k = top_k

    def forward(self, radar_map: torch.Tensor):
        B, T, R = radar_map.shape
        values, indices = torch.topk(radar_map, k=self.top_k, dim=-1)
        normalized_coords = (indices.float() / R) * 2.0 - 1.0
        normalized_coords = normalized_coords.reshape(B, T * self.top_k)
        intensities = values.reshape(B, T * self.top_k)
        return normalized_coords, intensities


class MultiModalRadarFrontend(nn.Module):
    MODALITY_NAMES = ["V1_Range", "V1_Doppler", "V2_Range", "V2_Doppler"]

    def __init__(self, top_k: int = 32, d_model: int = 256, num_modalities: int = 4):
        super().__init__()
        self.top_k = top_k
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.sparsifiers = nn.ModuleList(
            [ModalitySparsifier(top_k=top_k) for _ in range(num_modalities)]
        )
        self.modality_embeddings = nn.Embedding(num_modalities, 2)
        self.input_proj = nn.Linear(2, d_model)

    def forward(self, radar_input: torch.Tensor) -> torch.Tensor:
        B, num_mod, T, R = radar_input.shape
        token_list = []
        for mod_idx in range(self.num_modalities):
            single_map = radar_input[:, mod_idx, :, :]
            coords, intensities = self.sparsifiers[mod_idx](single_map)
            tokens = torch.stack([coords, intensities], dim=-1)  # [B, T*top_k, 2]
            mod_idx_tensor = torch.tensor(mod_idx, device=radar_input.device)
            mod_emb = self.modality_embeddings(mod_idx_tensor)   # [2]
            tokens = tokens + mod_emb
            token_list.append(tokens)
        all_tokens = torch.cat(token_list, dim=1)                # [B, 4*T*top_k, 2]
        mamba_ready = self.input_proj(all_tokens)                # [B, 4*T*top_k, d_model]
        return mamba_ready


# ──────────────────────────────────────────────────────────────
# 模块：MambaEncoder
# 职责：堆叠 N 层 Mamba block，对序列做全局时序建模
#
# Mamba block 超参说明（d_model=256 时的推荐值）：
#   d_state  = 16   : SSM 隐状态维度，控制"记忆容量"，16 是官方默认值
#   d_conv   = 4    : 局部卷积核大小，用于在 SSM 前做短程上下文聚合
#   expand   = 2    : 内部扩展比，实际内部维度 = d_model * expand = 512
#                     参数量 ≈ 3 * d_model^2 * expand，256×2 约 400K/层
# ──────────────────────────────────────────────────────────────
class MambaEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv:  int = 4,
        expand:  int = 2,
        num_layers: int = 4,
    ):
        """
        :param d_model:    特征维度，必须与 Frontend 的 d_model 严格一致
        :param d_state:    SSM 隐状态维度
        :param d_conv:     局部卷积核大小
        :param expand:     内部扩展比
        :param num_layers: Mamba block 层数
        """
        super().__init__()

        # 堆叠 num_layers 个 Mamba block
        # 每层之间加 LayerNorm，与 Mamba 官方实现的 pre-norm 风格一致
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),           # Pre-norm
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )
            for _ in range(num_layers)
        ])

        # 最终输出的 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, seq_len, d_model]，来自 MultiModalRadarFrontend 的输出
        :return:  [B, seq_len, d_model]，经过全局序列建模后的特征
        """
        for layer in self.layers:
            # 残差连接：x = x + Mamba(LayerNorm(x))
            # 与 Transformer 的 pre-norm 残差结构完全对齐
            x = x + layer(x)

        x = self.final_norm(x)
        return x


# ──────────────────────────────────────────────────────────────
# 模块：PureMambaBackbone
# 职责：将 Frontend 和 Encoder 串联为完整骨干网络
# ──────────────────────────────────────────────────────────────
class PureMambaBackbone(nn.Module):
    def __init__(
        self,
        top_k:      int = 32,
        d_model:    int = 256,
        d_state:    int = 16,
        d_conv:     int = 4,
        expand:     int = 2,
        num_layers: int = 4,
    ):
        super().__init__()
        self.frontend = MultiModalRadarFrontend(top_k=top_k, d_model=d_model)
        self.encoder  = MambaEncoder(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_layers,
        )

    def forward(self, radar_input: torch.Tensor) -> torch.Tensor:
        """
        :param radar_input: [B, 4, T, R]
        :return:            [B, seq_len, d_model]
        """
        x = self.frontend(radar_input)   # [B, seq_len, d_model]
        x = self.encoder(x)              # [B, seq_len, d_model]
        return x


# ──────────────────────────────────────────────────────────────
# 单元测试
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    BATCH      = 2
    T          = 16
    R          = 256
    TOP_K      = 32
    D_MODEL    = 256
    NUM_LAYERS = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")

    # 模拟输入
    dummy_input = torch.rand(BATCH, 4, T, R).to(device)

    # 实例化骨干网络
    backbone = PureMambaBackbone(
        top_k=TOP_K,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
    ).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"骨干网络总参数量: {total_params / 1e6:.2f} M")

    # 前向传播
    with torch.no_grad():
        output = backbone(dummy_input)

    # 验证
    expected_seq = 4 * T * TOP_K  # 2048
    print("=" * 50)
    print(f"输入形状:  {dummy_input.shape}")
    print(f"输出形状:  {output.shape}")
    assert output.shape == (BATCH, expected_seq, D_MODEL), "形状验证失败！"
    print(f"形状验证通过 ✓")

    # 显存占用（仅 CUDA 下有效）
    if device.type == "cuda":
        mem_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
        print(f"当前显存占用: {mem_mb:.1f} MB")
    print("=" * 50)