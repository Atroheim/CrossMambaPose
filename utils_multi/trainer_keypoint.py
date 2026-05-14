"""Training process
"""  
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
import torch.nn as nn
from torch.autograd import Variable
import tqdm
from utils_multi.result_utils import *
from omegaconf import OmegaConf

class Trainer:

    def __init__(self, model, data_train, data_valid, data_test, data_test_video, args, device):
        self.model = model
        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        self.data_test_video = data_test_video
        self.args = args
        self.device = device

        # --- TensorBoard 初始化 ---
        log_path = os.path.join(os.getcwd(), "tensorboard_logs")
        self.writer = SummaryWriter(log_dir=log_path)
        print(f"TensorBoard 日志将保存至: {log_path}")

    def train(self):
        self.model = self.model.to(self.device)
        loss_fn = nn.MSELoss().to(self.device)
        loss_fn_leg = nn.MSELoss().to(self.device)
        loss_fn_hand = nn.MSELoss().to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.args.train.learning_rate, weight_decay=self.args.train.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                T_0 = 25,
                                                T_mult = 1,
                                                eta_min = 0.01*self.args.train.learning_rate)

        Epoch_num = self.args.train.epoch
        step = 0
        for epoch in range(Epoch_num):
            self.model.train()
            progress_bar = tqdm.tqdm(self.data_train)
            for iter, (x_batch, x_R_batch, y_batch) in enumerate(progress_bar):
                x_batch = Variable(x_batch.float().to(self.device))
                x_R_batch = Variable(x_R_batch.float().to(self.device))
                y_batch = Variable(y_batch.float().to(self.device))

                y_batch_pred = self.model(x_batch, x_R_batch)
                loss_coord = loss_fn(y_batch_pred.to(dtype=torch.float32), y_batch.to(dtype=torch.float32))
                loss_leg = loss_fn_leg(y_batch_pred[:,:,(2,3,5,6),:], y_batch[:,:,(2,3,5,6),:])
                loss_hand = loss_fn_hand(y_batch_pred[:,:,(12,13,15,16),:], y_batch[:,:,(12,13,15,16),:])
                loss_motion_leg = motion_cal(y_batch_pred[:,:,(2,3,5,6),:], y_batch[:,:,(2,3,5,6),:], intervals=[2,4,6,8])
                loss_motion_hand = motion_cal(y_batch_pred[:,:,(12,13,15,16),:], y_batch[:,:,(12,13,15,16),:], intervals=[2,4,6,8])

                loss = loss_coord + (loss_leg+loss_hand)*self.args.train.alpha_limb + (loss_motion_leg+loss_motion_hand)*self.args.train.alpha_limb_motion

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                progress_bar.set_description(
                'Step: {}. Epoch: {}/{}. Total loss: {:.3f}. Coord Loss: {:.3f}. Motion Loss_Leg: {:.3f}. Motion Loss_Hand: {:.3f}'.
                format(step, epoch+1, Epoch_num, loss.item(), loss_coord.item(), loss_motion_leg.item()*0.05, loss_motion_hand.item()*0.05))

                # --- 核心修改：记录训练 Loss (按 step 记录) ---
                self.writer.add_scalar('Train/Total_Loss', loss.item(), step)
                self.writer.add_scalar('Train/Coord_Loss', loss_coord.item(), step)
                self.writer.add_scalar('Train/Motion_Leg_Loss', loss_motion_leg.item(), step)

            # 每个 Epoch 跑完进行测试
            test_loss, des_test = test_keypoint(self.data_test, self.device, self.model, output_temporal=True)

            # --- 核心修改：记录测试指标 (按 epoch 记录) ---
            self.writer.add_scalar('Test/MPJPE', test_loss['MPJPE'].mean(), epoch + 1)
            self.writer.add_scalar('Test/PCK', test_loss['PCK'].mean() * 100, epoch + 1)
            self.writer.add_scalar('Test/PCC', test_loss['PCC'][:, 1:].mean(), epoch + 1)
            self.writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch + 1)

            print('test_MPJPE: {:.3f}. test_PCC: {:.3f}. test_PCK: {:.3f}%'.
                format(test_loss['MPJPE'].mean(), test_loss['PCC'][:,1:].mean(), test_loss['PCK'].mean()*100))
            # --- 核心修改：智能保存最高分模型 ---
            current_pck = test_loss['PCK'].mean() * 100
            if not hasattr(self, 'best_pck'):
                self.best_pck = 0.0
            
            if current_pck > self.best_pck:
                self.best_pck = current_pck
                save_path = os.path.join(os.getcwd(), 'best_model.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f"🚀 发现更高精度模型 (PCK: {current_pck:.3f}%)！已成功保存至: {save_path}")
            # ------------------------------------

            if self.args.wandb.use_wandb:
                wandb.log({
                    'lr': lr_scheduler.optimizer.param_groups[0]['lr'],
                    'test_MPJPE': test_loss['MPJPE'].mean(),
                    'test_PCK': test_loss['PCK'].mean()*100,
                    })

            lr_scheduler.step()

        # 训练结束关闭 writer
        self.writer.close()