"""
train_mamba.py
Pure-Mamba 3D HPE — 完整训练脚本 (真实数据接入版)

数据流确认（来自 dataloader_multi.py + transform链反推）：
  DataLoader batch[0]: x_mD   → [B, 2, 128, 512]       # mD 双视角
  DataLoader batch[1]: x_Rng  → [B, 2, 128, 128]       # Rng 双视角
  DataLoader batch[2]: target → [B, 16, 17, 3]         # 3D GT，16帧×17关节×xyz

  模型输出：main_Net(x_mD, x_Rng) → [B, 16, 17, 3]

运行方式（正式模式）：
  conda activate mambapose && python train_mamba.py
"""

import os
import types
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. 超参数（集中管理，对应 Hydra YAML 字段）
# ─────────────────────────────────────────────────────────────
def build_args():
    args = types.SimpleNamespace()

    # ── 模型

    args.model = types.SimpleNamespace(
        dim=128,
        depth=4,
        cross_depth=2,
        # patch_size 是显存核心旋钮：L = (Dop_size/ph)×(win_size/pw)
        # [2,2]  → L=16384/视角  OOM
        # [4,16] → L=1024/视角   下一阶段目标
        # [8,16] → L=512/视角    当前值，先跑通Loss闭环
        patch_size=[8, 16],
        d_state=16,
        dropout=0.1,
        encoder_input='dual',
    )

    # ── 数据变换（已严格对齐真实 YAML 物理刻度）
    args.transforms = types.SimpleNamespace(
        Dop_size=128,          # 真实 mD 图高（Doppler bins）
        win_size=512,          # 真实 mD 图宽（时间帧）
        R_size_rng=128,        # 真实 Rng 图高（Range bins）
        win_size_rng=128,      # 真实 Rng 图宽（时间帧）
        win_sec=3.,            # 真实时间窗口
        test_ndiv=10,          # 真实测试切分
        radar1_mean=2.8967,    
        radar2_mean=2.9696,
        radar1_std=2.4292,
        radar2_std=2.3260,
        radar1_rng_mean=43.9563,
        radar2_rng_mean=44.2900,
        radar1_rng_std=3.2869,
        radar2_rng_std=3.3434,
    )

    # ── 训练
    args.train = types.SimpleNamespace(
        batch_size=16,     # patch[8,16]下约8GB显存，稳定后可升至32
        num_workers=4,
        lr=1e-4,
        weight_decay=1e-4, 
        epochs=1000,
        warmup_epochs=5,        # Cosine LR warmup 轮数
        traintest_split='random',
        traintest_class='all',
        checkpoint_dir='./checkpoints/Exp-A_NoCrossGate',
        log_dir='./runs/Exp-A_NoCrossGate',
        save_every=10,
        sgdr_T0=100,
        sgdr_T_mult=2,
        resume='./checkpoints/Exp-A_NoCrossGate/best.pth',
        alpha_limb=0.5,          # 四肢精确MSE权重（原版默认）
        alpha_limb_motion=0.05,  # 四肢运动连续性权重（原版默认）
    )

    # ── 数据路径
    args.result = types.SimpleNamespace(
        csv_file='/root/autodl-tmp/dataset/des_all.csv',
        data_dir='/root/autodl-tmp/dataset',
    )
    
    class _Preprocess:
        load = False      
        def keys(self):
            return []     
    args.preprocess = _Preprocess()

    return args


# ─────────────────────────────────────────────────────────────
# 2. 损失函数
# ─────────────────────────────────────────────────────────────
class MPJPELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_joint_error = torch.norm(pred - target, dim=-1)   # [B, T, J]
        return per_joint_error.mean()


def nPCC_loss(output: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    完全复制原版 nPCC_loss（result_utils.py 第259行）。
    输入：[B, T, J, 3]（沿 dim=1 即 T 轴计算相关性）
    输出：[B, J, 3]，值域 [0, 2]，越小越好
    原版：沿 dim=1 计算，1 - Pearson_r
    """
    target_mean = target.mean(dim=1, keepdim=True)   # [B, 1, J, 3]
    output_mean = output.mean(dim=1, keepdim=True)
    Pcc = torch.sum((target - target_mean) * (output - output_mean), dim=1) /           ((torch.sqrt(torch.sum((target - target_mean)**2, dim=1) + eps)) *
           (torch.sqrt(torch.sum((output - output_mean)**2, dim=1) + eps)))
    return 1 - Pcc   # [B, J, 3]


def motion_cal(
    predicted: torch.Tensor,
    target:    torch.Tensor,
    intervals: list = [2, 4, 6, 8],
) -> torch.Tensor:
    """
    完全复制原版 motion_cal（result_utils.py 第267行）。

    输入：[B, T, J, 3]
    原版用 torch.cross(a, b, dim=3) 计算间隔帧的运动方向叉积，
    再用 nPCC_loss 强制预测叉积方向与GT方向相关。

    物理意义：
      叉积 = 两帧关节位移向量的法向量，代表运动平面的法线方向
      1 - PCC(叉积) → 强制预测的运动方向与GT运动方向一致
      intervals=[2,4,6,8] → 同时监督短程(2帧)和长程(8帧)运动连续性

    注意：原版输入是 [B, N_div, T, J, 3]，dim=3 是 J 维。
    本函数输入是 [B, T, J, 3]，torch.cross 的 dim 对应 J 维 = dim=2。
    """
    assert predicted.shape == target.shape
    loss = torch.tensor(0.0, device=predicted.device)
    for itv in intervals:
        # [B, T-itv, J, 3]：间隔 itv 帧的两个位置向量
        pred_encode   = torch.cross(predicted[:, :-itv], predicted[:, itv:], dim=-1)
        target_encode = torch.cross(target[:, :-itv],    target[:, itv:],    dim=-1)
        loss = loss + torch.mean(nPCC_loss(pred_encode, target_encode)) / len(intervals)
    return loss


class MultiTaskLoss(nn.Module):
    """
    完整复现原版多任务损失：
      loss = loss_coord
           + (loss_leg + loss_hand) * alpha_limb
           + (loss_motion_leg + loss_motion_hand) * alpha_limb_motion

    COCO 17关节索引：
      腿部末端 (2,3,5,6)  → 左膝(2)、右膝(3)、左踝(5)、右踝(6)
      手臂末端 (12,13,15,16) → 左肘(12)、右肘(13)、左腕(15)、右腕(16)

    alpha 参考原版 YAML 默认值（可在 build_args 里调整）
    """
    # COCO 关节索引（与原版完全对齐）
    LEG_IDX  = [2, 3, 5, 6]       # 左膝、右膝、左踝、右踝
    HAND_IDX = [12, 13, 15, 16]   # 左肘、右肘、左腕、右腕

    def __init__(self, alpha_limb: float = 0.5, alpha_limb_motion: float = 0.05):
        """
        :param alpha_limb:        四肢精确 MSE 的权重（原版默认 0.5）
        :param alpha_limb_motion: 四肢运动连续性损失的权重（原版默认 0.05）
        """
        super().__init__()
        self.alpha_limb        = alpha_limb
        self.alpha_limb_motion = alpha_limb_motion
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred / target: [B, T, J=17, 3]
        返回：(total_loss, loss_coord, loss_motion_leg, loss_motion_hand)
              后三项用于 TensorBoard 分项监控
        """
        # 1. 全身坐标 MSE（主损失）
        loss_coord = self.mse(pred, target)

        # 2. 四肢精确监督（对末端关节加权）
        loss_leg  = self.mse(pred[:, :, self.LEG_IDX,  :],
                             target[:, :, self.LEG_IDX,  :])
        loss_hand = self.mse(pred[:, :, self.HAND_IDX, :],
                             target[:, :, self.HAND_IDX, :])

        # 3. 四肢运动连续性（叉积方向一致性）
        loss_motion_leg  = motion_cal(pred[:, :, self.LEG_IDX,  :],
                                      target[:, :, self.LEG_IDX,  :])
        loss_motion_hand = motion_cal(pred[:, :, self.HAND_IDX, :],
                                      target[:, :, self.HAND_IDX, :])

        # 4. 加权合并（与原版公式完全一致）
        total = (loss_coord
                 + (loss_leg + loss_hand) * self.alpha_limb
                 + (loss_motion_leg + loss_motion_hand) * self.alpha_limb_motion)

        return total, loss_coord, loss_motion_leg, loss_motion_hand


# ─────────────────────────────────────────────────────────────
# 3. 评估指标：PCK
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_pck(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.2) -> float:
    torso_size = torch.norm(target[:, :, 5, :] - target[:, :, 12, :], dim=-1) # [B, T]
    per_joint_error = torch.norm(pred - target, dim=-1)       # [B, T, J]
    correct = (per_joint_error < threshold * torso_size.unsqueeze(-1)).float()
    return correct.mean().item() * 100.0


# ─────────────────────────────────────────────────────────────
# 4. 学习率调度：SGDR（CosineAnnealingWarmRestarts）
# ─────────────────────────────────────────────────────────────
def build_scheduler(optimizer, args, is_resume: bool = False):
    """
    对齐原版 MVDoppler-Pose 的余弦退火热重启策略（SGDR）。

    周期设计（T_0=100, T_mult=2）：
      Cycle1: 100epoch  LR: max→min
      Cycle2: 200epoch  LR: max→min
      Cycle3: 400epoch  LR: max→min

    续训时（is_resume=True）跳过 warmup，直接用 SGDR，
    防止 SequentialLR 的 epoch 计数与续训 epoch 不兼容导致 LR 异常。
    """
    sgdr = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.train.sgdr_T0,
        T_mult=args.train.sgdr_T_mult,
        eta_min=1e-6,
    )
    if is_resume:
        # 续训：直接返回 SGDR，LR 从 max_lr 重新开始第一个周期
        return sgdr

    # 首次训练：5个epoch warmup后接SGDR
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=args.train.warmup_epochs,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, sgdr],
        milestones=[args.train.warmup_epochs],
    )
# ─────────────────────────────────────────────────────────────
# 5. 单 Epoch 训练
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_pck  = 0.0
    n_batches  = 0

    for batch_idx, batch in enumerate(loader):
        x_mD   = batch[0].to(device, non_blocking=True).float()   # [B, 2, 128, 512]
        x_Rng  = batch[1].to(device, non_blocking=True).float()   # [B, 2, 128, 128]
        target = batch[2].to(device, non_blocking=True).float()   # [B, 16, 17, 3]

        optimizer.zero_grad(set_to_none=True)

        pred = model(x_mD, x_Rng)
        # MultiTaskLoss 返回 (total, coord, motion_leg, motion_hand)
        loss, loss_coord, loss_motion_leg, loss_motion_hand = criterion(pred, target)

        # NaN 检测：跳过异常 batch，防止权重污染
        if torch.isnan(loss) or torch.isinf(loss):
            log.warning(f"[Train] Epoch {epoch:03d} Batch {batch_idx:04d}: Loss={loss.item()}, 跳过")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_pck  += compute_pck(pred.detach(), target.detach())
        n_batches  += 1

        if batch_idx % 20 == 0:
            log.info(
                f"[Train] Epoch {epoch:03d} | Batch {batch_idx:04d}/{len(loader):04d} | "
                f"Loss {loss.item():.4f} Coord {loss_coord.item():.4f} "
                f"MotionLeg {loss_motion_leg.item():.4f} MotionHand {loss_motion_hand.item():.4f}"
            )

    return total_loss / n_batches, total_pck / n_batches


# ─────────────────────────────────────────────────────────────
# 6. 辅助指标
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_pcc_ndiv(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    完全复制原版 MVDoppler-Pose result_utils.py 的 PCC，逐行对应：

      target_mean = target.mean(dim=2).unsqueeze(dim=2)
      output_mean = output.mean(dim=2).unsqueeze(dim=2)
      Pcc = sum((t-t_mean)*(p-p_mean), dim=2) / (sqrt(...)*sqrt(...))
      Pcc.mean(dim=3) → [B, N_frame, 3]（对J=17取均值）

    trainer 报告：test_loss['PCC'][:, 1:].mean()
      dim=3是坐标轴C，[:, 1:]跳过第0个crop窗口(N_frame维)，
      对剩余9个窗口和3个坐标轴取均值。

    输入：pred/target [B, N_frame, T, J=17, C=3]
    """
    eps = 1e-6
    target_mean = target.mean(dim=2).unsqueeze(dim=2)   # [B, N, 1, J, 3]
    output_mean = pred.mean(dim=2).unsqueeze(dim=2)

    Pcc = torch.sum((target - target_mean) * (pred - output_mean), dim=2) / \
          ((torch.sqrt(torch.sum((target - target_mean)**2, dim=2) + eps)) *
           (torch.sqrt(torch.sum((pred   - output_mean)**2, dim=2) + eps)))
    # Pcc: [B, N_frame, J, C]

    Pcc_mean = Pcc.mean(dim=3)              # [B, N_frame, J]（对J取均值）
    return Pcc_mean[:, 1:].mean().item()    # 跳过第0窗口，与原版trainer对齐

@torch.no_grad()
def compute_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> float:
    """归一化坐标 MPJPE，直接对标 MVDoppler-Pose Test/MPJPE=0.0674"""
    return torch.norm(pred - target, dim=-1).mean().item()




# ─────────────────────────────────────────────────────────────
# 7. 验证
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    返回 (avg_loss, avg_pck, avg_mpjpe, avg_pcc)

    PCC 计算说明：
      test batch 含 n_div 维度：[B, n_div, 2, H, W]
      模型推理：展平 n_div → B*n_div，得 pred [B*n_div, T, J, 3]
      PCC：reshape 回 [B, n_div, T, J, 3]，严格对齐原版 [B, N_frame, T, 17, 3]
      train batch（ndim=4）：n_div=1，unsqueeze 补维度
    """
    model.eval()
    total_loss  = 0.0
    total_pck   = 0.0
    total_mpjpe = 0.0
    total_pcc   = 0.0
    n_batches   = 0

    for batch in loader:
        x_mD_raw  = batch[0].to(device, non_blocking=True).float()
        x_Rng_raw = batch[1].to(device, non_blocking=True).float()
        tgt_raw   = batch[2].to(device, non_blocking=True).float()

        if x_mD_raw.ndim == 5:
            # test: [B, n_div, C, H, W]
            B, n_div, C, H_mD, W_mD = x_mD_raw.shape
            _, _,    _, H_Rng, W_Rng = x_Rng_raw.shape
            _, _,    T, J, xyz       = tgt_raw.shape
            # 展平 n_div 送模型
            x_mD  = x_mD_raw.view(B * n_div, C, H_mD, W_mD)
            x_Rng = x_Rng_raw.view(B * n_div, C, H_Rng, W_Rng)
            tgt   = tgt_raw.view(B * n_div, T, J, xyz)
        else:
            # train: [B, C, H, W]，n_div=1
            B, n_div = x_mD_raw.shape[0], 1
            T, J, xyz = tgt_raw.shape[1], tgt_raw.shape[2], tgt_raw.shape[3]
            x_mD, x_Rng, tgt = x_mD_raw, x_Rng_raw, tgt_raw

        pred = model(x_mD, x_Rng)   # [B*n_div, T, J, 3]

        total_loss  += criterion(pred, tgt)[0].item()  # MultiTaskLoss 返回元组，取 total
        total_pck   += compute_pck(pred, tgt)
        total_mpjpe += compute_mpjpe(pred, tgt)

        # PCC：reshape 回 [B, n_div, T, J, 3]，对齐原版 N_frame 维度
        pred_nd = pred.view(B, n_div, T, J, xyz)
        tgt_nd  = tgt.view(B, n_div, T, J, xyz)
        total_pcc += compute_pcc_ndiv(pred_nd, tgt_nd)

        n_batches += 1

    return (
        total_loss  / n_batches,
        total_pck   / n_batches,
        total_mpjpe / n_batches,
        total_pcc   / n_batches,
    )


# ─────────────────────────────────────────────────────────────
# 7. Checkpoint 工具
# ─────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, epoch, best_pck, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'epoch':      epoch,
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict(),
        'best_pck':   best_pck,
    }, path)
    log.info(f"Checkpoint 保存: {path}")

def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    log.info(f"断点续训，从 Epoch {ckpt['epoch']} 恢复，历史最优 PCK={ckpt['best_pck']:.2f}%")
    return ckpt['epoch'] + 1, ckpt['best_pck']


# ─────────────────────────────────────────────────────────────
# 8. 主函数
# ─────────────────────────────────────────────────────────────
def main():
    args   = build_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"运行设备: {device}")

    # ── 数据集：接入真实 DataLoader
    from utils_multi.dataloader_multi import LoadDataset_Keypoint
    data_train, data_test, _, (n_train, n_test) = LoadDataset_Keypoint(args)
    log.info(f"训练集 {n_train} 样本，测试集 {n_test} 样本")

    # ── 模型：接入已替换核心的 Mamba 引擎
    from model.ablation_B_no_gate import main_Net
    model = main_Net(args).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"模型总参数量: {total_params / 1e6:.3f} M")

    # ── 形状验证
    log.info("执行前向传播形状验证...")
    with torch.no_grad():
        _b = next(iter(data_train))
        # 根据 batch 返回值直接验证真实数据连通性
        _out = model(_b[0].to(device).float(), _b[1].to(device).float())
        assert _out.shape == (args.train.batch_size, 16, 17, 3) or _out.shape == (_b[0].shape[0], 16, 17, 3), \
            f"输出形状异常: {_out.shape}"
    log.info(f"真实数据形状验证通过 ✓  输出: {list(_out.shape)}")

    # ── 损失 + 优化器 + 调度器
    criterion = MultiTaskLoss(
        alpha_limb=args.train.alpha_limb,
        alpha_limb_motion=args.train.alpha_limb_motion,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
    )
    is_resume = bool(args.train.resume and os.path.isfile(args.train.resume))
    scheduler = build_scheduler(optimizer, args, is_resume=is_resume)
    writer    = SummaryWriter(log_dir=args.train.log_dir)

    # ── 断点续训
    start_epoch = 0
    best_pck    = 0.0
    if args.train.resume and os.path.isfile(args.train.resume):
        start_epoch, best_pck = load_checkpoint(
            model, optimizer, scheduler, args.train.resume, device
        )

    # ── 训练主循环
    log.info(f"开始训练，共 {args.train.epochs} Epochs")
    for epoch in range(start_epoch, args.train.epochs):

        train_loss, train_pck = train_one_epoch(
            model, data_train, optimizer, criterion, device, epoch
        )
        val_loss, val_pck, val_mpjpe, val_pcc = evaluate(model, data_test, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        log.info(
            f"Epoch {epoch:03d}/{args.train.epochs} | LR {current_lr:.2e} | "
            f"Train Loss {train_loss:.4f} PCK {train_pck:.2f}% | "
            f"Val Loss {val_loss:.4f} PCK {val_pck:.2f}% MPJPE {val_mpjpe:.4f} PCC {val_pcc:.4f}"
        )

        writer.add_scalar("Loss/train",  train_loss,  epoch)
        writer.add_scalar("Loss/val",    val_loss,    epoch)
        writer.add_scalar("PCK/train",   train_pck,   epoch)
        writer.add_scalar("PCK/val",     val_pck,     epoch)
        writer.add_scalar("MPJPE/val",   val_mpjpe,   epoch)
        writer.add_scalar("PCC/val",     val_pcc,     epoch)
        writer.add_scalar("LR",          current_lr,  epoch)

        if val_pck > best_pck:
            best_pck = val_pck
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_pck,
                path=os.path.join(args.train.checkpoint_dir, "best.pth"),
            )
            log.info(f"★ 新最优 Val PCK: {best_pck:.2f}% | MPJPE: {val_mpjpe:.4f} | PCC: {val_pcc:.4f}")

        if (epoch + 1) % args.train.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_pck,
                path=os.path.join(args.train.checkpoint_dir, f"epoch_{epoch:04d}.pth"),
            )

    writer.close()
    log.info(f"训练完成。最优 Val PCK: {best_pck:.2f}%")


if __name__ == "__main__":
    main()