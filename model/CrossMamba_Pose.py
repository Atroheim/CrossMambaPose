"""
CrossMamba_Pose.py — 时序对齐门控版（Temporal-Aligned Cross-Gating）

架构升级（消融实验 E vs B 的对比点）：
  旧版门控（全局均值）：
    global_mD = out_mD.mean(dim=1)          # [B, d]，丢失时序结构
    gate = sigmoid(Linear(global_mD))        # [B, 1, d]，所有帧用同一门控

  新版门控（时序对齐）：
    out_mD reshape → [B, T=16, P=16, d]     # 按帧拆分
    frame_mD = out_mD.mean(dim=2)           # [B, T=16, d]，每帧独立摘要
    gate = sigmoid(Linear(frame_mD))         # [B, T=16, 1, d]
    out_Rng reshape → [B, T=16, P=16, d]
    out_Rng_gated = out_Rng * gate           # 第t帧速度摘要门控第t帧距离特征

  物理动机（CFAR物理直觉）：
    人体在第t帧运动速度（mD能量）决定该帧距离谱的SNR
    快速运动帧→高SNR→放大距离特征；静止帧→低SNR→抑制噪声

  超参不变（控制变量，保证消融实验有效性）：
    d_model=256, num_layers=4, T=16, P=8, num_patches=8
    RadarPatchEmbed、时间池化、回归头完全不变
"""
import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba


# ──────────────────────────────────────────────────────────────
# 1. PatchEmbed（不变）
# ──────────────────────────────────────────────────────────────
class RadarPatchEmbed(nn.Module):
    def __init__(self, in_bins=128, patch_size=16, d_model=256):
        super().__init__()
        self.num_patches = in_bins // patch_size   # 8
        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, Bins = x.shape
        x = x.reshape(B * T, 1, Bins)
        x = self.proj(x)                            # [B*T, d_model, num_patches]
        x = x.transpose(1, 2).contiguous()         # [B*T, num_patches, d_model]
        x = self.norm(x)
        x = x.reshape(B, T * self.num_patches, -1) # [B, T*8=128, d_model]
        return x


# ──────────────────────────────────────────────────────────────
# 2. CrossMambaBlock — 时序对齐门控版（核心升级）
# ──────────────────────────────────────────────────────────────
class CrossMambaBlock(nn.Module):
    """
    时序对齐交叉门控（Temporal-Aligned Cross-Gating）。

    token 结构：每轨道 [B, 256, d] = [B, 2视角×16帧×8patch, d]
    门控计算：
      Step1: Mamba 全序列扫描（保持不变）
      Step2: reshape → [B, 16帧, 16token/帧, d]，按帧分组
      Step3: 帧内均值 → [B, 16, d]，每帧一个物理摘要向量
      Step4: Linear+Sigmoid → [B, 16, d] 门控权重
      Step5: unsqueeze → [B, 16, 1, d]，广播到帧内所有 token
      Step6: 对方轨道 reshape → [B, 16, 16, d]，逐帧门控调制
      Step7: reshape 回 [B, 256, d]，残差连接

    T_FRAMES=16, TOKENS_PER_FRAME=16（2视角×8patch）
    """
    T_FRAMES        = 16   # 时间帧数，与GT对齐
    TOKENS_PER_FRAME = 16  # 每帧的token数 = 2视角 × 8patch

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Mamba 扫描器（不变）
        self.mamba_Rng = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_mD  = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)

        # Pre-Norm（不变）
        self.norm_Rng = nn.LayerNorm(d_model)
        self.norm_mD  = nn.LayerNorm(d_model)

        # 时序对齐门控投影
        # 输入：帧级摘要 [B, T, d]；输出：帧级门控权重 [B, T, d]
        # 初始化：weight~N(0,0.01)，bias=0 → 初始gate≈0.5，不破坏早期梯度流
        self.gate_mD_to_Rng = nn.Linear(d_model, d_model)
        self.gate_Rng_to_mD = nn.Linear(d_model, d_model)
        nn.init.normal_(self.gate_mD_to_Rng.weight, 0, 0.01)
        nn.init.zeros_(self.gate_mD_to_Rng.bias)
        nn.init.normal_(self.gate_Rng_to_mD.weight, 0, 0.01)
        nn.init.zeros_(self.gate_Rng_to_mD.bias)

        # FFN（不变）
        self.ffn_Rng = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_mD = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def _temporal_gate(
        self,
        src: torch.Tensor,   # 门控来源 [B, 256, d]，提供门控权重
        tgt: torch.Tensor,   # 门控目标 [B, 256, d]，被调制
        gate_proj: nn.Linear,
    ) -> torch.Tensor:
        """
        时序对齐门控：用 src 的帧级摘要调制 tgt 的逐帧 token。

        :param src:       [B, T*P=256, d]，来源轨道（提供门控）
        :param tgt:       [B, T*P=256, d]，目标轨道（被调制）
        :param gate_proj: d→d 的线性投影
        :return:          [B, T*P=256, d]，调制后的目标特征
        """
        B = src.shape[0]
        T = self.T_FRAMES           # 16
        P = self.TOKENS_PER_FRAME   # 16

        # Step1: src reshape → [B, T, P, d]，按帧分组
        src_framed = src.view(B, T, P, self.d_model)          # [B, 16, 16, d]

        # Step2: 帧内均值 → [B, T, d]
        # 物理意义：把每帧内 16 个 token 的能量分布压缩为一个摘要向量
        frame_summary = src_framed.mean(dim=2)                 # [B, 16, d]

        # Step3: 生成帧级门控权重 [B, T, d]
        gate = torch.sigmoid(gate_proj(frame_summary))         # [B, 16, d]

        # Step4: unsqueeze → [B, T, 1, d]，广播到帧内所有 token
        gate = gate.unsqueeze(2)                               # [B, 16, 1, d]

        # Step5: tgt reshape → [B, T, P, d]，逐帧门控调制
        tgt_framed = tgt.view(B, T, P, self.d_model)          # [B, 16, 16, d]
        tgt_gated  = tgt_framed * gate                         # 广播：[B,16,16,d]*[B,16,1,d]

        # Step6: reshape 回序列格式
        return tgt_gated.view(B, T * P, self.d_model)          # [B, 256, d]

    def forward(self, x_Rng: torch.Tensor, x_mD: torch.Tensor):
        """
        x_Rng / x_mD: [B, 256, d_model]
        return:        x_Rng, x_mD: [B, 256, d_model]
        """
        # Step1: 独立 Mamba 序列建模（Pre-Norm 残差）
        out_Rng = x_Rng + self.mamba_Rng(self.norm_Rng(x_Rng))
        out_mD  = x_mD  + self.mamba_mD(self.norm_mD(x_mD))

        # Step2: 时序对齐交叉门控
        # mD 的帧级速度摘要 → 调制 Rng 的帧级距离特征（同帧对齐）
        out_Rng_gated = self._temporal_gate(out_mD,  out_Rng, self.gate_mD_to_Rng)
        out_mD_gated  = self._temporal_gate(out_Rng, out_mD,  self.gate_Rng_to_mD)

        # Step3: 残差连接
        x_Rng = x_Rng + out_Rng_gated
        x_mD  = x_mD  + out_mD_gated

        # Step4: FFN 残差
        x_Rng = x_Rng + self.ffn_Rng(x_Rng)
        x_mD  = x_mD  + self.ffn_mD(x_mD)

        return x_Rng, x_mD


# ──────────────────────────────────────────────────────────────
# 3. 主网络（不变）
# ──────────────────────────────────────────────────────────────
class main_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model     = 256
        self.num_layers  = 4
        self.time_frames = 16
        self.num_patches = 8

        self.pool_mD  = nn.AvgPool1d(kernel_size=32, stride=32)
        self.pool_Rng = nn.AvgPool1d(kernel_size=8,  stride=8)

        self.patch_embed_Rng = RadarPatchEmbed(in_bins=128, patch_size=16, d_model=self.d_model)
        self.patch_embed_mD  = RadarPatchEmbed(in_bins=128, patch_size=16, d_model=self.d_model)

        self.view_embeddings = nn.Embedding(2, self.d_model)

        self.cross_mamba_layers = nn.ModuleList([
            CrossMambaBlock(d_model=self.d_model) for _ in range(self.num_layers)
        ])

        self.final_norm_Rng = nn.LayerNorm(self.d_model)
        self.final_norm_mD  = nn.LayerNorm(self.d_model)

        self.frame_head = nn.Sequential(
            nn.Linear(self.d_model * 2, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 17 * 3),
        )

    def _time_compress(self, x: torch.Tensor, is_mD: bool) -> torch.Tensor:
        pool = self.pool_mD if is_mD else self.pool_Rng
        x = x.transpose(1, 2)
        x = pool(x)
        x = x.transpose(1, 2)
        return x

    def _process_modality(self, x, is_mD, view_idx):
        x = self._time_compress(x, is_mD)
        embed = self.patch_embed_mD if is_mD else self.patch_embed_Rng
        tokens = embed(x)
        view_emb = self.view_embeddings(torch.tensor(view_idx, device=x.device))
        return tokens + view_emb

    def _tokens_to_frames(self, tokens: torch.Tensor) -> torch.Tensor:
        B, total_seq, d = tokens.shape
        T, P, V = self.time_frames, self.num_patches, 2
        tokens = tokens.view(B, V, T, P, d)
        tokens = tokens.mean(dim=(1, 3))
        return tokens

    def forward(self, x_mD: torch.Tensor, x_R: torch.Tensor) -> torch.Tensor:
        x_mD = x_mD.transpose(-1, -2)
        x_R  = x_R.transpose(-1, -2)

        r1 = self._process_modality(x_R[:,  0], is_mD=False, view_idx=0)
        r2 = self._process_modality(x_R[:,  1], is_mD=False, view_idx=1)
        m1 = self._process_modality(x_mD[:, 0], is_mD=True,  view_idx=0)
        m2 = self._process_modality(x_mD[:, 1], is_mD=True,  view_idx=1)

        x_Rng = torch.cat([r1, r2], dim=1)   # [B, 256, d]
        x_mD  = torch.cat([m1, m2], dim=1)   # [B, 256, d]

        for layer in self.cross_mamba_layers:
            x_Rng, x_mD = layer(x_Rng, x_mD)

        x_Rng = self.final_norm_Rng(x_Rng)
        x_mD  = self.final_norm_mD(x_mD)

        feat_Rng = self._tokens_to_frames(x_Rng)
        feat_mD  = self._tokens_to_frames(x_mD)

        feat_per_frame = torch.cat([feat_Rng, feat_mD], dim=-1)
        out = self.frame_head(feat_per_frame)
        out = out.view(out.shape[0], self.time_frames, 17, 3).contiguous()
        return out


# ──────────────────────────────────────────────────────────────
# 单元测试
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import types
    args = types.SimpleNamespace()

    B = 2
    dummy_mD = torch.randn(B, 2, 128, 512).cuda()
    dummy_R  = torch.randn(B, 2, 128, 128).cuda()

    model = main_Net(args).cuda()

    with torch.no_grad():
        out = model(dummy_mD, dummy_R)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"输入 x_mD  : {list(dummy_mD.shape)}")
    print(f"输入 x_R   : {list(dummy_R.shape)}")
    print(f"输出形状   : {list(out.shape)}  ← [B, 16帧, 17关节, xyz]")
    print(f"总参数量   : {total/1e6:.3f} M")
    assert out.shape == (B, 16, 17, 3)
    print("形状验证通过 ✓")