"""
mamba_pose.py
Pure-Mamba 3D HPE 项目 —— 完整前向通路（骨干 + 姿态回归头）

完整结构：
  [B, 4, T, R]
      │
  MultiModalRadarFrontend   →  [B, 2048, 256]
      │
  MambaEncoder (4层)         →  [B, 2048, 256]
      │
  全局平均池化               →  [B, 256]
      │
  PoseRegressionHead (MLP)  →  [B, 17, 3]
      │
  输出：17个 COCO 关节的三维坐标 (x, y, z)

运行方式：
  conda activate mambapose && python mamba_pose.py
"""

import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba


# ──────────────────────────────────────────────────────────────
# 前端模块（与 mamba_modules.py 完全一致，复用）
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
            tokens = torch.stack([coords, intensities], dim=-1)
            mod_idx_tensor = torch.tensor(mod_idx, device=radar_input.device)
            mod_emb = self.modality_embeddings(mod_idx_tensor)
            tokens = tokens + mod_emb
            token_list.append(tokens)
        all_tokens = torch.cat(token_list, dim=1)
        mamba_ready = self.input_proj(all_tokens)
        return mamba_ready


# ──────────────────────────────────────────────────────────────
# 编码器（与 mamba_encoder.py 完全一致，复用）
# ──────────────────────────────────────────────────────────────
class MambaEncoder(nn.Module):
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        x = self.final_norm(x)
        return x


# ──────────────────────────────────────────────────────────────
# 模块：PoseRegressionHead
# 职责：将骨干输出的全局特征向量映射为 17 个 COCO 关节的三维坐标
#
# COCO 17关节顺序（索引 0-16）：
#   0:鼻子  1:左眼  2:右眼  3:左耳  4:右耳
#   5:左肩  6:右肩  7:左肘  8:右肘  9:左腕  10:右腕
#   11:左髋 12:右髋 13:左膝 14:右膝 15:左踝  16:右踝
# ──────────────────────────────────────────────────────────────
class PoseRegressionHead(nn.Module):
    def __init__(self, d_model: int = 256, num_joints: int = 17):
        """
        :param d_model:    输入特征维度，与骨干的 d_model 严格一致
        :param num_joints: 关节点数量，COCO 格式为 17
        """
        super().__init__()
        self.num_joints = num_joints

        # 两层 MLP：d_model → 512 → num_joints*3
        # 中间加 ReLU + Dropout，防止过拟合
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 512),          # 升维，增加表达能力
            nn.ReLU(),
            nn.Dropout(p=0.1),                # 轻度正则化
            nn.Linear(512, num_joints * 3),   # 输出 17×3=51 维
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 骨干编码器输出，形状 [B, seq_len, d_model]
        :return:  关节坐标，形状 [B, 17, 3]
        """
        # Step 1: 全局平均池化，沿序列维度取均值
        # [B, seq_len, d_model] → [B, d_model]
        # 物理意义：将 2048 个稀疏雷达点的全局动力学特征压缩为一个摘要向量
        x = x.mean(dim=1)

        # Step 2: MLP 回归关节坐标
        # [B, d_model] → [B, num_joints*3]
        x = self.mlp(x)

        # Step 3: reshape 为关节格式
        # [B, 51] → [B, 17, 3]，每行对应一个关节的 (x, y, z)
        x = x.view(x.shape[0], self.num_joints, 3)

        return x


# ──────────────────────────────────────────────────────────────
# 完整模型：PureMambaPose
# ──────────────────────────────────────────────────────────────
class PureMambaPose(nn.Module):
    def __init__(
        self,
        top_k:      int = 32,
        d_model:    int = 256,
        d_state:    int = 16,
        d_conv:     int = 4,
        expand:     int = 2,
        num_layers: int = 4,
        num_joints: int = 17,
    ):
        super().__init__()
        self.frontend = MultiModalRadarFrontend(top_k=top_k, d_model=d_model)
        self.encoder  = MambaEncoder(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand, num_layers=num_layers,
        )
        self.head = PoseRegressionHead(d_model=d_model, num_joints=num_joints)

    def forward(self, radar_input: torch.Tensor) -> torch.Tensor:
        """
        :param radar_input: [B, 4, T, R]
        :return:            [B, 17, 3]，17个 COCO 关节的三维坐标
        """
        x = self.frontend(radar_input)   # [B, seq_len, d_model]
        x = self.encoder(x)              # [B, seq_len, d_model]
        x = self.head(x)                 # [B, 17, 3]
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
    NUM_JOINTS = 17

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")

    # 模拟输入
    dummy_input = torch.rand(BATCH, 4, T, R).to(device)

    # 实例化完整模型
    model = PureMambaPose(
        top_k=TOP_K,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_joints=NUM_JOINTS,
    ).to(device)

    # 参数量统计（分模块显示）
    frontend_params = sum(p.numel() for p in model.frontend.parameters())
    encoder_params  = sum(p.numel() for p in model.encoder.parameters())
    head_params     = sum(p.numel() for p in model.head.parameters())
    total_params    = frontend_params + encoder_params + head_params

    print(f"Frontend 参数量: {frontend_params / 1e6:.3f} M")
    print(f"Encoder  参数量: {encoder_params  / 1e6:.3f} M")
    print(f"Head     参数量: {head_params     / 1e6:.3f} M")
    print(f"总参数量:        {total_params    / 1e6:.3f} M")

    # 前向传播
    with torch.no_grad():
        output = model(dummy_input)

    # 验证
    print("=" * 50)
    print(f"输入形状:  {dummy_input.shape}")
    print(f"输出形状:  {output.shape}")
    assert output.shape == (BATCH, NUM_JOINTS, 3), "形状验证失败！"
    print(f"形状验证通过 ✓  输出含义: [Batch, 17关节, (x,y,z)]")

    # 显存占用
    if device.type == "cuda":
        mem_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
        print(f"当前显存占用: {mem_mb:.1f} MB")
    print("=" * 50)