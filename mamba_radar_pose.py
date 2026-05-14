"""
mamba_radar_pose.py
Pure-Mamba 3D HPE — replaces MobileViT backbone entirely.

Architecture:
  Input:  x_mD  [B, 2, Dop_size, win_size]
          x_Rng [B, 2, R_size_rng, win_size_rng]
  ↓ PatchEmbed  → [B, 2, L, D]
  ↓ BiMamba per-view encoding
  ↓ Cross-Mamba view fusion  → [B, 2, L, D]
  ↓ Global AvgPool           → [B, 2, D]
  ↓ Concat mD + Rng          → [B, 4D]
  ↓ MLP regression head      → [B, 16*17*3]
  ↓ reshape                  → [B, 16, 17, 3]

Dependencies:
  pip install mamba-ssm einops
"""

import math
import torch
import torch.nn as nn
from einops import rearrange

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("请先安装: pip install mamba-ssm")


# ─────────────────────────────────────────────
# 1. Patch Embedding  [B, 2, H, W] → [B, 2, L, D]
# ─────────────────────────────────────────────
class PatchEmbed(nn.Module):
    """
    把 2D 雷达图线性展平成 token 序列。
    patch_size=(ph, pw) 决定 token 粒度，L = (H/ph)*(W/pw)。
    两个视角共享同一个投影权重。
    """
    def __init__(self, in_ch: int, dim: int, patch_size=(2, 2)):
        super().__init__()
        self.ph, self.pw = patch_size
        self.proj = nn.Linear(in_ch * patch_size[0] * patch_size[1], dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, 2, H, W]
        B, V, H, W = x.shape
        x = rearrange(x, 'b v (h ph) (w pw) -> b v (h w) (ph pw)',
                      ph=self.ph, pw=self.pw)          # [B, 2, L, ph*pw]
        x = self.norm(self.proj(x))                    # [B, 2, L, D]
        return x


# ─────────────────────────────────────────────
# 2. BiMamba Block  — 双向扫描单元
# ─────────────────────────────────────────────
class BiMambaBlock(nn.Module):
    """
    前向 + 反向各一个 Mamba，输出相加后过 LayerNorm + FFN。
    d_state / d_conv / expand 是 Mamba 内部超参，保持默认即可。
    """
    def __init__(self, dim: int, d_state=16, d_conv=4, expand=2, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, L, D]
        residual = x
        x = self.norm1(x)
        fwd = self.mamba_fwd(x)
        bwd = self.mamba_bwd(x.flip(1)).flip(1)
        x = residual + fwd + bwd          # 双向求和，保持维度
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 3. Cross-Mamba Block  — 双视角交叉扫描
# ─────────────────────────────────────────────
class CrossMambaBlock(nn.Module):
    """
    参考 VMamba 交叉扫描思路：
      视角1 query 从视角2 的序列中提取上下文（反向），反之亦然。

    具体实现：
      concat([v1, v2]) 沿 L 维 → 送入 Mamba → 取前半段更新 v1
      concat([v2, v1]) 沿 L 维 → 送入 Mamba → 取前半段更新 v2
    两次 forward 共享同一组 Mamba 权重（weight-tied cross scan）。
    """
    def __init__(self, dim: int, d_state=16, d_conv=4, expand=2, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # 共享权重的交叉 Mamba
        self.cross_mamba = Mamba(d_model=dim, d_state=d_state,
                                 d_conv=d_conv, expand=expand)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, v1, v2):
        # v1, v2: [B, L, D]
        L = v1.shape[1]

        # v1 attending to v2
        cat12 = torch.cat([self.norm1(v1), self.norm1(v2)], dim=1)  # [B, 2L, D]
        out12 = self.cross_mamba(cat12)[:, :L, :]                    # 取前半段
        v1 = v1 + out12
        v1 = v1 + self.ffn1(v1)

        # v2 attending to v1
        cat21 = torch.cat([self.norm2(v2), self.norm2(v1)], dim=1)
        out21 = self.cross_mamba(cat21)[:, :L, :]
        v2 = v2 + out21
        v2 = v2 + self.ffn2(v2)

        return v1, v2


# ─────────────────────────────────────────────
# 4. 单模态 Mamba 编码器
# ─────────────────────────────────────────────
class MambaEncoder(nn.Module):
    """
    对一个模态的双视角特征做 per-view BiMamba 编码。
    输入 [B, 2, L, D]，输出 [B, 2, L, D]。
    """
    def __init__(self, dim: int, depth: int, **mamba_kwargs):
        super().__init__()
        # 两个视角共享同一组 BiMamba 权重（可改为独立）
        self.layers = nn.ModuleList([
            BiMambaBlock(dim, **mamba_kwargs) for _ in range(depth)
        ])

    def forward(self, x):
        # x: [B, 2, L, D]
        B, V, L, D = x.shape
        x = rearrange(x, 'b v l d -> (b v) l d')
        for layer in self.layers:
            x = layer(x)
        x = rearrange(x, '(b v) l d -> b v l d', b=B, v=V)
        return x


# ─────────────────────────────────────────────
# 5. 顶层网络 main_Net
# ─────────────────────────────────────────────
class main_Net(nn.Module):
    """
    纯 Mamba 双模态雷达姿态估计网络。

    Args (来自 Hydra args):
      args.model.dim          : Mamba 隐层维度，默认 128
      args.model.depth        : BiMamba 层数，默认 4
      args.model.cross_depth  : CrossMamba 层数，默认 2
      args.model.patch_size   : patch 尺寸 (ph, pw)，默认 (2,2)
      args.model.d_state      : Mamba d_state，默认 16
      args.transforms.Dop_size / win_size        : mD 分支图像尺寸
      args.transforms.R_size_rng / win_size_rng  : Rng 分支图像尺寸
    """
    def __init__(self, args):
        super().__init__()

        dim         = getattr(args.model, 'dim', 128)
        depth       = getattr(args.model, 'depth', 4)
        cross_depth = getattr(args.model, 'cross_depth', 2)
        patch_size  = tuple(getattr(args.model, 'patch_size', [2, 2]))
        d_state     = getattr(args.model, 'd_state', 16)
        dropout     = getattr(args.model, 'dropout', 0.1)

        mamba_kwargs = dict(d_state=d_state, d_conv=4, expand=2, dropout=dropout)

        # ── Patch Embedding（每个模态各自一个，因 H/W 不同）
        self.embed_mD  = PatchEmbed(in_ch=1, dim=dim, patch_size=patch_size)
        self.embed_Rng = PatchEmbed(in_ch=1, dim=dim, patch_size=patch_size)

        # ── Per-view BiMamba 编码
        self.encoder_mD  = MambaEncoder(dim, depth, **mamba_kwargs)
        self.encoder_Rng = MambaEncoder(dim, depth, **mamba_kwargs)

        # ── Cross-Mamba 视角融合（每个模态内部，视角1↔视角2）
        self.cross_mD  = nn.ModuleList([
            CrossMambaBlock(dim, **mamba_kwargs) for _ in range(cross_depth)
        ])
        self.cross_Rng = nn.ModuleList([
            CrossMambaBlock(dim, **mamba_kwargs) for _ in range(cross_depth)
        ])

        # ── 全局池化后拼接：mD(2*dim) + Rng(2*dim) → 4*dim
        fusion_dim = 4 * dim

        # ── 回归头 → [B, 16*17*3]
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, 16 * 17 * 3),
            nn.Tanh(),
        )

        self._init_weights()

    # ── 权重初始化
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── 单模态前向：embed → BiMamba → CrossMamba → pool → [B, 2*D]
    def _encode_modal(self, x, embed, encoder, cross_layers):
        # x: [B, 2, H, W]
        B, V, H, W = x.shape
        # 逐视角 patch embed，共享权重
        v1 = embed(x[:, 0:1, :, :].expand(B, 1, H, W))   # [B, 1, L, D] — in_ch=1
        # 注意 PatchEmbed in_ch=1，所以每次送一张图
        # 先把 [B,2,H,W] 拆成 [B,1,H,W] x 2
        v1 = embed(x[:, [0], :, :])   # [B, 1, L, D]
        v2 = embed(x[:, [1], :, :])   # [B, 1, L, D]
        # stack → [B, 2, L, D]
        feat = torch.stack([v1.squeeze(1), v2.squeeze(1)], dim=1)

        # Per-view BiMamba
        feat = encoder(feat)                               # [B, 2, L, D]

        # Cross-Mamba
        cv1 = feat[:, 0]   # [B, L, D]
        cv2 = feat[:, 1]
        for layer in cross_layers:
            cv1, cv2 = layer(cv1, cv2)

        # Global avg pool → [B, D] each → concat → [B, 2D]
        out = torch.cat([cv1.mean(dim=1), cv2.mean(dim=1)], dim=-1)
        return out  # [B, 2*D]

    def forward(self, x_mD, x_Rng):
        """
        x_mD  : [B, 2, Dop_size, win_size]
        x_Rng : [B, 2, R_size_rng, win_size_rng]
        return : [B, 16, 17, 3]
        """
        feat_mD  = self._encode_modal(x_mD,  self.embed_mD,  self.encoder_mD,  self.cross_mD)
        feat_Rng = self._encode_modal(x_Rng, self.embed_Rng, self.encoder_Rng, self.cross_Rng)

        # 跨模态拼接
        feat = torch.cat([feat_mD, feat_Rng], dim=-1)   # [B, 4*D]

        # 回归
        out = self.head(feat)                            # [B, 16*17*3]
        out = rearrange(out, 'b (t j c) -> b t j c',
                        t=16, j=17, c=3).contiguous()   # [B, 16, 17, 3]
        return out


# ─────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import types

    def make_args():
        args = types.SimpleNamespace()
        args.model = types.SimpleNamespace(
            dim=128, depth=4, cross_depth=2,
            patch_size=[2, 2], d_state=16, dropout=0.1
        )
        args.transforms = types.SimpleNamespace(
            Dop_size=64,  win_size=16,
            R_size_rng=32, win_size_rng=16,
        )
        return args

    args = make_args()
    model = main_Net(args).cuda()

    B = 2
    x_mD  = torch.randn(B, 2, args.transforms.Dop_size,  args.transforms.win_size).cuda()
    x_Rng = torch.randn(B, 2, args.transforms.R_size_rng, args.transforms.win_size_rng).cuda()

    with torch.no_grad():
        out = model(x_mD, x_Rng)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem   = torch.cuda.memory_allocated() / 1024**2

    print("=" * 50)
    print(f"输入 x_mD  : {list(x_mD.shape)}")
    print(f"输入 x_Rng : {list(x_Rng.shape)}")
    print(f"输出形状   : {list(out.shape)}  ← [B, 16帧, 17关节, xyz]")
    print(f"总参数量   : {total/1e6:.3f} M")
    print(f"当前显存   : {mem:.1f} MB")
    print("=" * 50)
    assert out.shape == (B, 16, 17, 3), "形状验证失败！"
    print("形状验证通过 ✓")