import torch
import hydra
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # 💡 引入 numpy 用于保存文件
from tqdm import tqdm
from scipy import signal
from utils_multi.result_utils import *
from utils_multi.camera import *
from utils_multi import dataloader_multi

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


from model import ablation_C_global_gate 

def result_load(device, path_model, path_args):
  args = OmegaConf.load(path_args)

  if 'encoder_input' not in args.model.keys():
    args.model.encoder_input = 'multi'
  args.train.traintest_split = 'subject_independent'
  data_train, data_test, data_test_video, len_data = dataloader_multi.LoadDataset_Keypoint(args)
  

  model = ablation_C_global_gate.main_Net(args).to(device)

  keypoint_startpoint = [data_test_video.__getitem__(i)[2][1] for i in range(len(data_test_video))]
  list_episode = [data_test_video.__getitem__(i)[4] for i in range(len(data_test_video))]


  checkpoint = torch.load(path_model, map_location=device)
  if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
  else:
    state_dict = checkpoint


  new_state_dict = {}
  for k, v in state_dict.items():
    name = k.replace('module.', '').replace('model.', '')
    new_state_dict[name] = v

  model.load_state_dict(new_state_dict, strict=False)
  model.eval()
  model_size = count_parameters(model)

  test_loss, des_test, (y_target, y_pred) = test_keypoint(data_test, device, model, output_pred=True, output_temporal=True)
  class_idx = [i for i in range(len(des_test))]
  test_loss_sel = {}
  for key in test_loss.keys():
    test_loss_sel[key] = test_loss[key][class_idx]
  des_test_sel = [des_test[idx] for idx in class_idx]

  return test_loss_sel, des_test_sel, (y_target[class_idx], y_pred[class_idx]), model_size, args, keypoint_startpoint, list_episode

@hydra.main(version_base=None, config_path="conf", config_name="config_inference")
def main(args: DictConfig) -> None:
  config = OmegaConf.to_container(args)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  path_save = config['path_save']
  FPS = 90
  
  list_keypoint_start = [16, 36, 56, 76, 96, 116, 136, 156, 176, 196]

  test_loss, des_test, (y_target, y_pred), _, _, _, _ = result_load(device, config['path_model'], config['path_args'])
  num_sample, num_tstart   = y_pred.shape[0], y_pred.shape[1]
  for i_sample in tqdm(range(num_sample)):
    name_episode  = des_test[i_sample]['fname'].split('.')[0]
    name_subject  = des_test[i_sample]['subject']
    name_pattern  = des_test[i_sample]['pattern']
    if str(args.test_episode) in name_episode:
      show3Dpose_fromradar_video(device, y_target[i_sample], list_keypoint_start, select_frame=5, path=f'{path_save}/{name_episode}-{name_subject}-{name_pattern}-GT.mp4')
      show3Dpose_fromradar_video(device, y_pred[i_sample], list_keypoint_start, select_frame=5, path=f'{path_save}/{name_episode}-{name_subject}-{name_pattern}-Pred.mp4')   
      
      gt_data = y_target[i_sample]
      pred_data = y_pred[i_sample]
      if hasattr(gt_data, 'cpu'):
        gt_data = gt_data.cpu().numpy()
      if hasattr(pred_data, 'cpu'):
        pred_data = pred_data.cpu().numpy()

      np.save(f'./{name_episode}-GT_full.npy', gt_data)
      np.save(f'./{name_episode}-Ours.npy', pred_data)


if __name__ == '__main__':
  main()