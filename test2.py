from __future__ import print_function
import numpy as np
import numpy.ma as ma
import json
import time
import sys
from datetime import datetime
import pathlib
import shutil
import yaml
from argparse import ArgumentParser
import os
from functools import partial
from sklearn import metrics
from tqdm import tqdm, trange
import torchvision.models as models

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from models.SUNet18 import SUNet18
from models.MTBIT import MTBIT, MTBIT_FPN4, MTBIT_FPN4_DIFF,MTBIT_FPN5_DIFF,MTBIT_FPN5 , MTBIT_FPN3 , MTBIT_FPN2, MTBIT_FPN4_ATTEN, MTBIT_FPN5_ATTEN, MTBIT_FPN4_abs ##########my
from models.ChangeFormer import ChangeFormerV1, ChangeFormerV2, ChangeFormerV3, ChangeFormerV4, ChangeFormerV5, ChangeFormerV6   ##########my
from models.DTCDSCN import CDNet34  ##########my

from dataloader import Dataset
from augmentations import get_validation_augmentations, get_training_augmentations
from losses import choose_criterion3d, choose_criterion2d
from optim import set_optimizer, set_scheduler
from cp import pretrain_strategy

from PIL import Image
import tifffile as tiff
from matplotlib import pyplot

# 解决问题：TypeError: Object of type MaskedArray is not JSON serializable
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ma.MaskedArray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)

def get_args():
    parser = ArgumentParser(description = "Hyperparameters", add_help = True)
    parser.add_argument('-c', '--config-name', type = str, help = 'YAML Config name', dest = 'CONFIG', default = 'config')
    parser.add_argument('-nw', '--num-workers', type = str, help = 'Number of workers', dest = 'num_workers', default = 2)
    parser.add_argument('-v', '--verbose', type = bool, help = 'Verbose validation metrics', dest = 'verbose', default = False)

    parser.add_argument('--output_folder', default='./predict/3dcdmy200', type=str)

    return parser.parse_args()

# to calculate rmse
def metric_mse(inputs, targets, mask, exclude_zeros = False):
    if exclude_zeros:
        mask_ = mask.copy()
        indices_one = mask_ == 1
        indices_zero = mask_ == 0
        mask_[indices_one] = 0 # replacing 1s with 0s
        mask_[indices_zero] = 1 # replacing 0s with 1s
        inputs = ma.masked_array(inputs, mask=mask_)
        targets = ma.masked_array(targets, mask=mask_)
        loss = (inputs - targets) ** 2
        n_pixels = np.count_nonzero(targets)
        return np.sum(loss)/n_pixels
    else:
        loss = (inputs - targets) ** 2
        return np.mean(loss)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())

def save_image_2D(image_numpy, image_path):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    
    image_pil = Image.fromarray(np.array(image_numpy,dtype=np.uint8))
    image_pil.save(image_path)

# def save_image_3D(image_numpy, image_path):
#     """Save a numpy image to the disk
#     Parameters:
#         image_numpy (numpy array) -- input numpy array
#         image_path (str)          -- the path of the image
#     """
    
#     image_pil = Image.fromarray(np.array(image_numpy,dtype=np.uint8))
#     image_pil.save(image_path)
    
args = get_args()

device = 'cuda'
cuda = True
num_GPU = 1
torch.cuda.set_device(0)
manual_seed = 18
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

config_name = args.CONFIG
# config_name = 'try'
config_path = './config/'+config_name
default_dst_dir = "./results/3dcdmy200/"
out_file = default_dst_dir + config_name + '/'
predict_file_2D = args.output_folder + '/'+ config_name + '/' + '2D' + '/'
predict_file_3D = args.output_folder + '/'+ config_name + '/' + '3D' + '/'
predict_file_3D_pred = predict_file_3D + 'pred'
predict_file_3D_label = predict_file_3D + 'label3d'
os.makedirs(out_file, exist_ok=True)
os.makedirs(predict_file_2D, exist_ok=True)
os.makedirs(predict_file_3D_pred, exist_ok=True)
os.makedirs(predict_file_3D_label, exist_ok=True)

# Load the configuration params of the experiment
full_config_path = config_path + ".yaml"
print(f"Loading experiment {full_config_path}")
with open(full_config_path, "r") as f:
    exp_config = yaml.load(f, Loader=yaml.SafeLoader)

print(f"Logs and/or checkpoints will be stored on {out_file}")
shutil.copyfile(full_config_path, out_file+'config.yaml')
print("Config file correctly saved!")

stats_file = open(out_file + 'stats.txt', 'a', buffering=1)
print(' '.join(sys.argv), file=stats_file)
print(' '.join(sys.argv))

print(exp_config)
print(exp_config, file=stats_file)

x_train_dir = exp_config['data']['train']['path']
x_valid_dir = exp_config['data']['val']['path']
x_test_dir = exp_config['data']['test']['path']

batch_size = exp_config['data']['train']['batch_size']

lweight2d, lweight3d = exp_config['model']['loss_weights']
weights2d = exp_config['model']['2d_loss_weights']

augmentation = exp_config['data']['augmentations']
min_scale = exp_config['data']['min_value']
max_scale = exp_config['data']['max_value']

mean = exp_config['data']['mean']
std = exp_config['data']['std']

if augmentation:
    train_transform = get_training_augmentations(m = mean, s = std)
else:
  train_transform = get_validation_augmentations(m = mean, s = std)

valid_transform = get_validation_augmentations(m = mean, s = std)

train_dataset = Dataset(x_train_dir,
                        augmentation = train_transform)

valid_dataset = Dataset(x_valid_dir,
                        augmentation = valid_transform)
                        
test_dataset = Dataset(x_test_dir,
                        augmentation = valid_transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

name_3dloss = exp_config['model']['3d_loss']
exclude_zeros = exp_config['model']['exclude_zeros'] 
criterion3d = choose_criterion3d(name = name_3dloss)

class_weights2d = torch.FloatTensor(weights2d).to(device)
name_2dloss = exp_config['model']['2d_loss'] 
criterion2d = choose_criterion2d(name_2dloss, class_weights2d) #, class_ignored)

nepochs = exp_config['optim']['num_epochs']
lr = exp_config['optim']['lr']

model = exp_config['model']['model']
classes = exp_config['model']['num_classes']

pretrain = exp_config['model']['pretraining_strategy']
arch = exp_config['model']['feature_extractor_arch']
CHECKPOINTS = exp_config['model']['checkpoints_path']

encoder, pretrained, _ = pretrain_strategy(pretrain, CHECKPOINTS, arch)

if model == "SUNet18":
    net = SUNet18(3, 2, resnet = encoder).to(device)
elif model == 'mtbit_resnet18':
  net = MTBIT(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=1, dec_depth=8, decoder_dim_head=16).to(device)
elif model == 'mtbit_resnet50':
  net = MTBIT(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=16, backbone = 'resnet50').to(device)

elif model == 'ChangeFormer':
  net = ChangeFormerV6(input_nc=3, output_nc=2, img_size=256).to(device)

elif model == 'DTCDSCN':
  #The implementation of the paper"Building Change Detection for Remote Sensing Images Using a Dual Task Constrained Deep Siamese Convolutional Network Model "
  #Code copied from: https://github.com/fitzpchao/DTCDSCN
  net = CDNet34(in_channels=3).to(device)


elif model == 'ResVIT_FPN5_DIFF_e4d4':
  net = MTBIT_FPN5_DIFF(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=5, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device)

elif model == 'ResVIT_FPN4_DIFF_e4d4':
  net = MTBIT_FPN4_DIFF(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device)  

elif model == 'ResVIT_FPN4_abs_e4d4':
  net = MTBIT_FPN4_abs(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device)   


elif model == 'ResVIT_FPN5_e4d4':
  net = MTBIT_FPN5(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=5, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device)   

elif model == 'ResVIT_FPN3_e4d4':
  net = MTBIT_FPN3(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=3, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device)

elif model == 'ResVIT_FPN2_e4d4':
  net = MTBIT_FPN2(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=2, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device)

elif model == 'ResVIT_FPN4_e4d4':
  net = MTBIT_FPN4(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device)

elif model == 'ResVIT_FPN4_e2d6':
  net = MTBIT_FPN4(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=2, dec_depth=6, decoder_dim_head=8).to(device)

elif model == 'ResVIT_FPN4_e6d2':
  net = MTBIT_FPN4(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=6, dec_depth=2, decoder_dim_head=8).to(device)

elif model == 'ResVIT_FPN4_e1d8':
  net = MTBIT_FPN4(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=1, dec_depth=8, decoder_dim_head=8).to(device)

elif model == 'ResVIT_FPN4_e8d1':
  net = MTBIT_FPN4(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=8, dec_depth=1, decoder_dim_head=8).to(device)

elif model == 'ResVIT_FPN4_e4d4_CBAM':
  net = MTBIT_FPN4_ATTEN(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device)

elif model == 'ResVIT_FPN4_e4d4_CBAMECA':
  net = MTBIT_FPN4_ATTEN(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device)

elif model == 'ResVIT_FPN4_e4d4_SE':
  net = MTBIT_FPN4_ATTEN(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device)

elif model == 'ResVIT_FPN4_e4d4_SK':
  net = MTBIT_FPN4_ATTEN(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device)

elif model == 'ResVIT_FPN4_e4d4_ECA':
  net = MTBIT_FPN4_ATTEN(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device)        

elif model == 'ResVIT_FPN4_e4d4_ECA_ddm16':
  net = MTBIT_FPN4_ATTEN(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=16).to(device)     

elif model == 'ResVIT_FPN5_e4d4_ECA':
  net = MTBIT_FPN5_ATTEN(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=5, if_upsample_2x=True,
              enc_depth=4, dec_depth=4, decoder_dim_head=8).to(device) 
              
else:
	print('Model not implemented yet')

print('Model selected: ', model)

optimizer = set_optimizer(exp_config['optim'], net)
print('Optimizer selected: ', exp_config['optim']['optim_type'])
lr_adjust = set_scheduler(exp_config['optim'], optimizer)
print('Scheduler selected: ', exp_config['optim']['lr_schedule_type'])

res_cp = exp_config['model']['restore_checkpoints']
if os.path.exists(out_file+f'{res_cp}bestnet.pth'):
  net.load_state_dict(torch.load(out_file+f'{res_cp}bestnet.pth'))
  print('Checkpoints successfully loaded!')
else:
  print('No checkpoints founded')

tr_par, tot_par = count_parameters(net)
print(f'Trainable parameters: {tr_par}, total parameters {tot_par}')
print(f'Trainable parameters: {tr_par}, total parameters {tot_par}', file=stats_file)

net.eval()

TN = 0
FP = 0
FN = 0
TP = 0
mean_mae = 0
rmse1 = 0
rmse2 = 0

res_2d= []
res_3d= []
im_couple= []
names = []

for t1, t2, mask2d, mask3d, name in tqdm(test_loader): 


  t1 = t1.to(device)
  t2 = t2.to(device)

  #out2d bchw(1, 2, 200, 200)    out3d  bchw(1, 1, 200, 200)
  out2d, out3d = net(t1, t2)  
  
  out2d = out2d.detach().argmax(dim=1)  #bhw(1,200,200)
  out2d = out2d.cpu().numpy()
  out3d = out3d.detach().cpu().numpy()  #bchw (1, 1, 200, 200)
  out3d = (out3d*(max_scale - min_scale)+(max_scale+min_scale))/2

  mask3d = mask3d.cpu().numpy()

  im_couple.append([t1.cpu().squeeze().permute((1,2,0)), t2.cpu().squeeze().permute((1,2,0))])  #hwc (200,200,3)
  res_2d.append([out2d.squeeze(), mask2d.squeeze()])  #hw(200,200)  [[res2d,mask2d]]
  res_3d.append([out3d.squeeze(), mask3d.squeeze()])  #hw(200,200)  [[res3d,mask3d]]
  names.append(name)
  # print('res_3d.shape:', np.array(res_3d).shape)
  # img = res_3d[0][0]
  # mask = res_3d[0][1]
  # print('img.shape:', np.array(img).shape)
  # print('mask.shape:', np.array(mask).shape)
  # print(1/0)

  try:
      tn, fp, fn, tp = metrics.confusion_matrix(mask2d.ravel(), out2d.ravel()).ravel()
  except: 
      tn, fp, fn, tp = [0,0,0,0]
      print('Only 0 mask')


  mean_ae = metrics.mean_absolute_error(mask3d.ravel(), out3d.ravel())
  s_rmse1 = metric_mse(out3d.ravel(), mask3d.ravel(), mask2d.cpu().numpy().ravel(), exclude_zeros = False)
  s_rmse2 = metric_mse(out3d.ravel(), mask3d.ravel(), mask2d.cpu().numpy().ravel(), exclude_zeros = True)
  max_error = metrics.max_error(mask3d.ravel(), out3d.ravel())
  mask_max = np.abs(mask3d).max()
  
  if args.verbose:
    print()
    print(f'2D Val: TN: {tn},\tFN: {fn},\tTP: {tp},\tFP: {fp},\tF1 Score: {f1_score},\tIoU: {IoU}')
    print(f'3D Val: Mean Absolute Error: {mean_ae}, \tRMSE Error: {s_rmse}, \tMax Error: {max_error} (w.r.t {mask_max})')


  mean_mae += mean_ae
  rmse1 += s_rmse1
  rmse2 += s_rmse2
  TN += tn
  FP += fp
  FN += fn 
  TP += tp

for i in range(len(res_3d)):
  ####### 2d预测结果保存
  img_name = names[i][0]
  file_name_2d = os.path.join(predict_file_2D, img_name) + '.png'
  save_image_2D(255* res_2d[i][0], file_name_2d)

  ####### 3d预测结果保存
  # print(len(res_3d))
  file_name_3d_pred = os.path.join(predict_file_3D_pred, img_name) + '.png'
  file_name_3d_label = os.path.join(predict_file_3D_label, img_name) + '.png'
  #####预测图
  tiff.imshow(res_3d[i][0],vmin=-25, vmax=30)
  pyplot.savefig(file_name_3d_pred) 
  # pyplot.show()
  ### 3dlabel图
  # tiff.imshow(res_3d[i][1],vmin=-25, vmax=30)
  # pyplot.savefig(file_name_3d_label) 
  # pyplot.show()

#   ####### 3d预测结果保存
#   file_name_3d = os.path.join(predict_file_3D, img_name) + '.png'
# #   figure_3d, subplot_3d, image_3d = tiff.imshow(res_3d[0][0],vmin=-25, vmax=30)
#   pyplot.imsave(file_name_3d, res_3d[i][1], vmin=-25, vmax=30)


mean_mae = mean_mae/len(test_loader)
acc = (TP+TN)/(TP+FP+TN+FN)
mean_f1 = 2*TP/(2*TP+FP+FN)
mIoU = TP/(TP+FN+FP)
RMSE1 = np.sqrt(rmse1/len(test_loader))
RMSE2 = np.sqrt(rmse2/len(test_loader))

print(f'Test metrics - 2D: OA -> {acc*100} %; F1 Score -> {mean_f1*100} %; mIoU -> {mIoU*100} %; 3D: MAE -> {mean_mae} m; RMSE -> {RMSE1} m; cRMSE -> {RMSE2} m')
stats = dict(epoch = 'Test', MeanAbsoluteError = mean_mae, RMSE = RMSE1, cRMSE = RMSE2, OA  = acc*100, F1Score = mean_f1*100, mIoU = mIoU*100)
# print(json.dumps(stats), file=stats_file)
print(json.dumps(stats, cls = MyEncoder), file=stats_file)
