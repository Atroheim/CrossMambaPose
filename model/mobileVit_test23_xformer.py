"""
mobileVit_test23_xformer.py (已重构为 Pure-Mamba 3D 架构)

核心修改（v2）：
  原版问题：top_k=16，T=512（win_size经transpose后），总seq_len=32768，
            信息极度稀疏（每帧仅16个点），PCK上限~72%。
  修复方案：在 Sparsifier 前插入 AvgPool1d(kernel=32) 做时间维压缩，
            512帧 → 16帧（与GT的16帧对齐），top_k提升到32，
            总seq_len = 4×16×32 = 2048，信息密度提升2倍，显存安全。
"""
import math
import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba


# ──────────────────────────────────────────────────────────────
# 核心组件 1：物理坐标提取器 (Sparsifier)
# 输入：[B, T_compressed, R]
# 输出：normalized_coords [B, T*top_k], intensities [B, T*top_k]
# ──────────────────────────────────────────────────────────────
class ModalitySparsifier(nn.Module):
    def __init__(self, top_k: int = 32):
        super().__init__()
        self.top_k = top_k

    def forward(self, radar_map: torch.Tensor):
        B, T, R = radar_map.shape                              # [B, T_compressed, R]
        values, indices = torch.topk(radar_map, k=self.top_k, dim=-1)
        normalized_coords = (indices.float() / R) * 2.0 - 1.0 # [-1, 1]，消除量纲差异
        normalized_coords = normalized_coords.reshape(B, T * self.top_k)
        intensities = values.reshape(B, T * self.top_k)
        return normalized_coords, intensities


# ──────────────────────────────────────────────────────────────
# 主网络
# ──────────────────────────────────────────────────────────────
class main_Net(nn.Module):
    def __init__(self, args):
        super().__init__()

        # ── 时间维压缩参数
        # 输入经 transpose 后：x_mD [B, 2, T=512, R=128]
        # AvgPool1d 沿 T 维：512 → 512/32 = 16（与 GT 16帧对齐）
        # 物理意义：32个相邻时间帧能量平均，保留慢变动作包络，滤除帧间噪声
        self.time_pool_kernel = 32   # 512 / 32 = 16
        self.T_compressed     = 16   # 压缩后时间 token 数

        # ── Sparsifier 参数
        # seq_len = num_modalities × T_compressed × top_k
        #         = 4 × 16 × 32 = 2048  （与 Phase-1 MVP 完全一致，显存安全）
        self.top_k           = 64
        self.d_model         = 256
        self.num_layers      = 4
        self.num_modalities  = 4
        self.time_frames     = 16    # 输出帧数，与 GT 对齐

        # 1. 时间维压缩池化（无可学习参数，4个模态共用同一个实例）
        #    作用对象：[B*2*4, T=512, R] 中的 T 维
        #    使用 AvgPool1d：输入期望 [N, C, L]，此处 N=B, C=R, L=T
        #    → 需在调用处先 transpose 为 [B, R, T] 再 pool 再 transpose 回来
        self.time_pool = nn.AvgPool1d(
            kernel_size=self.time_pool_kernel,
            stride=self.time_pool_kernel,
        )

        # 2. 独立 Sparsifier（独立实例，便于后续扩展独立 BN）
        self.sparsifiers = nn.ModuleList([
            ModalitySparsifier(top_k=self.top_k) for _ in range(self.num_modalities)
        ])

        # 3. 物理模态嵌入（让 Mamba 区分距离 token 和速度 token）
        self.modality_embeddings = nn.Embedding(self.num_modalities, 2)

        # 4. 升维投影：2维物理坐标 → d_model
        self.input_proj = nn.Linear(2, self.d_model)

        # 5. Pure-Mamba 序列引擎（Pre-Norm 残差风格）
        self.mamba_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.d_model),
                Mamba(d_model=self.d_model, d_state=16, d_conv=4, expand=2)
            )
            for _ in range(self.num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)

        # 6. 3D 骨架回归头
        self.regression_head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.SiLU(),
            nn.Linear(512, self.time_frames * 17 * 3)  # 16*17*3=816
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _time_compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        对单模态雷达图做时间维压缩。
        :param x: [B, T, R]，T=512，R=128
        :return:  [B, T_compressed, R]，T_compressed=16
        """
        # AvgPool1d 期望 [B, C, L]，此处 C=R，L=T
        # [B, T, R] → [B, R, T] → pool → [B, R, T/32] → [B, T/32, R]
        x = x.transpose(1, 2)                      # [B, R, T]
        x = self.time_pool(x)                      # [B, R, T_compressed]
        x = x.transpose(1, 2)                      # [B, T_compressed, R]
        return x

    def forward(self, x_mD, x_R):
        """
        x_mD: [B, 2, Dop_size=128, win_size=512]  mD 双视角
        x_R:  [B, 2, R_size=128,   win_size=512]  Rng 双视角
        return: [B, 16, 17, 3]
        """
        # Step 1: CNN格式 → 序列格式
        # [B, 2, H, W] → [B, 2, T=W=512, R=H]
        # 物理意义：win_size(时间) 是序列轴，Dop/Range bins 是特征轴
        x_mD = x_mD.transpose(-1, -2)             # [B, 2, 512, 128]
        x_R  = x_R.transpose(-1, -2)              # [B, 2, 512, 128]

        # Step 2: 解耦 4 个物理通道
        mod_0 = x_R[:,  0, :, :]   # [B, 512, 128]  视角1 Range
        mod_1 = x_mD[:, 0, :, :]  # [B, 512, 128]  视角1 Doppler
        mod_2 = x_R[:,  1, :, :]   # [B, 512, 128]  视角2 Range
        mod_3 = x_mD[:, 1, :, :]  # [B, 512, 128]  视角2 Doppler
        mods  = [mod_0, mod_1, mod_2, mod_3]

        # Step 3: 时间维压缩 512→16，再 Top-K 稀疏化
        # 最终 seq_len = 4 × 16 × 32 = 2048
        token_list = []
        for mod_idx in range(self.num_modalities):
            # 时间压缩：[B, 512, 128] → [B, 16, 128]
            m = self._time_compress(mods[mod_idx])         # [B, T_compressed, R]

            # Top-K 稀疏化：每帧保留 top_k 个最强 bin
            coords, intensities = self.sparsifiers[mod_idx](m)
            # coords: [B, T_compressed*top_k]  intensities: [B, T_compressed*top_k]

            # 打包为 2D 物理 token
            tokens = torch.stack([coords, intensities], dim=-1)  # [B, T*K, 2]

            # 注入模态嵌入（让 Mamba 区分 Range 和 Doppler）
            mod_emb = self.modality_embeddings(
                torch.tensor(mod_idx, device=x_mD.device)
            )                                              # [2]
            tokens = tokens + mod_emb                     # 广播加

            token_list.append(tokens)

        # Step 4: 4模态拼接 → [B, 2048, 2]
        all_tokens = torch.cat(token_list, dim=1)

        # Step 5: 升维投影 → [B, 2048, d_model=256]
        x = self.input_proj(all_tokens)

        # Step 6: Mamba 全局序列建模
        for layer in self.mamba_layers:
            x = x + layer(x)                              # Pre-Norm 残差
        x = self.final_norm(x)

        # Step 7: 全局时序池化 → [B, d_model]
        global_feat = x.mean(dim=1)

        # Step 8: MLP 回归 → [B, 16, 17, 3]
        out = self.regression_head(global_feat)            # [B, 816]
        out = rearrange(out, 'b (t j c) -> b t j c',
                        t=16, j=17, c=3).contiguous()
        return out


# ──────────────────────────────────────────────────────────────
# 参数统计与测试
# ──────────────────────────────────────────────────────────────
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    import types
    args = types.SimpleNamespace()

    # 模拟真实 DataLoader 输出形状
    B = 2
    dummy_mD = torch.randn(B, 2, 128, 512).cuda()
    dummy_R  = torch.randn(B, 2, 128, 512).cuda()

    model = main_Net(args).cuda()
    out   = model(dummy_mD, dummy_R)

    print(f"输入 x_mD : {list(dummy_mD.shape)}")
    print(f"输入 x_R  : {list(dummy_R.shape)}")
    print(f"输出形状  : {list(out.shape)}  ← [B, 16帧, 17关节, xyz]")
    print(f"总参数量  : {count_parameters(model)/1e6:.3f} M")
    print(f"seq_len   : 4×{model.T_compressed}×{model.top_k} = {4*model.T_compressed*model.top_k}")
    assert out.shape == (B, 16, 17, 3), "形状验证失败"
    print("形状验证通过 ✓")