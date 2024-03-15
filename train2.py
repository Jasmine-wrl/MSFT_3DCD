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
    # parser.add_argument('-nw', '--num-workers', type = str, help = 'Number of workers', dest = 'num_workers', default = 2)
    parser.add_argument('-nw', '--num-workers', type = str, help = 'Number of workers', dest = 'num_workers', default = 0)  ####my
    parser.add_argument('-v', '--verbose', type = bool, help = 'Verbose validation metrics', dest = 'verbose', default = False)
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

args = get_args()

device = 'cuda'
cuda = True
num_GPU = 1
torch.cuda.set_device(0)
manual_seed = 18
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

config_name = args.CONFIG
config_path = './config/'+config_name
default_dst_dir = "./results/3dcdmy200/"  ########my
out_file = default_dst_dir + config_name + '/'
os.makedirs(out_file, exist_ok=True)

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

start = time.time()

best2dmetric = 0
best3dmetric = 1000000
  
net.train()

y_loss_2d = []
y_loss_3d = []
y_loss_total = []
y_OA = []
y_F1 = []
y_mIoU = []
y_MAE = []
y_RMSE = []
y_cRMSE = []

for epoch in range(1, nepochs):
  tot_2d_loss = 0
  tot_3d_loss = 0

  for param_group in optimizer.param_groups:
    print("Epoch: %s" % epoch, " - Learning rate: ", param_group['lr'])

  for t1, t2, mask2d, mask3d, name in tqdm(train_loader):

    t1 = t1.to(device)
    t2 = t2.to(device)
    
    mask3d = mask3d.to(device).float()
    out2d, out3d = net(t1, t2)
    
    if args.verbose:
        print()
        print('MASK 3D: ', torch.min(mask3d).item(), torch.max(mask3d).item())
        
    mask3d = 2*(mask3d - min_scale)/(max_scale - min_scale)-1
    
    if args.verbose:
        print('MASK 3D NORM: ', torch.min(mask3d).item(), torch.max(mask3d).item())
        print('OUT 3D: ', torch.min(out3d).item(), torch.max(out3d).item())

    loss2d = criterion2d(out2d, mask2d.to(device).long()) #long
    loss3d = criterion3d(out3d.squeeze(dim=1), mask3d) #, exclude_zeros = exclude_zeros)

    loss = lweight2d*loss2d + lweight3d*loss3d #sommo le loss

    optimizer.zero_grad()
    loss.backward() #backward delle loss
    optimizer.step()

    tot_2d_loss += loss2d.detach().cpu().numpy()*batch_size 
    tot_3d_loss += loss3d.detach().cpu().numpy()*batch_size

  epoch_2d_loss = tot_2d_loss/len(train_dataset)
  epoch_3d_loss = tot_3d_loss/len(train_dataset)
  epoch_loss = lweight2d*epoch_2d_loss + lweight3d*epoch_3d_loss

  lr_adjust.step()
  
  epoch_2d_loss = round(epoch_2d_loss, 5)
  epoch_3d_loss = round(epoch_3d_loss, 5)
  epoch_loss = round(epoch_loss, 5)

  y_loss_2d.append(epoch_2d_loss) 
  y_loss_3d.append(epoch_3d_loss)
  y_loss_total.append(epoch_loss)
  print(f"Training loss: {epoch_loss},\t2D Loss: {epoch_2d_loss}, \t3D Loss: {epoch_3d_loss}")
  

  with torch.no_grad():
    net.eval()

    TN = 0
    FP = 0
    FN = 0
    TP = 0
    mean_mae = 0
    rmse1 = 0
    rmse2 = 0

    for t1, t2, mask2d, mask3d ,name in tqdm(valid_loader): 

      t1 = t1.to(device)
      t2 = t2.to(device)

      out2d, out3d = net(t1, t2)
      out2d = out2d.detach().argmax(dim=1).cpu().numpy()
      out3d = out3d.detach().cpu().numpy()
      out3d = ((out3d.ravel()+1)/2)*(max_scale-min_scale)+min_scale
     
      try:
          tn, fp, fn, tp = metrics.confusion_matrix(mask2d.ravel(), out2d.ravel()).ravel()
      except: 
          tn, fp, fn, tp = [0,0,0,0]
          print('Only 0 mask') 
      
    #   cm = metrics.confusion_matrix(mask2d.ravel(), out2d.ravel()).ravel()
    #   print(type(cm.tolist()))
    #   print(cm.tolist()) 
    #   tn, fp, fn, tp = cm.tolist()
      
      mean_ae = metrics.mean_absolute_error(mask3d.ravel(), out3d.ravel())
      s_rmse1 = metric_mse(out3d.ravel(), mask3d.cpu().numpy().ravel(), mask2d.cpu().numpy().ravel(), exclude_zeros = False)
      s_rmse2 = metric_mse(out3d.ravel(), mask3d.cpu().numpy().ravel(), mask2d.cpu().numpy().ravel(), exclude_zeros = True)
      max_error = metrics.max_error(mask3d.ravel(), out3d.ravel())
      mask_max = np.abs(mask3d.cpu().numpy()).max()    
      
      mean_mae += mean_ae
      rmse1 += s_rmse1
      rmse2 += s_rmse2
      TN += tn
      FP += fp
      FN += fn 
      TP += tp
      
    mean_mae = mean_mae/len(valid_loader)
    acc = (TP+TN)/(TP+FP+TN+FN)  ###########my
    mIoU = TP/(TP+FN+FP)
    mean_f1 = 2*TP/(2*TP+FP+FN)
    RMSE1 = np.sqrt(rmse1/len(valid_loader))
    RMSE2 = np.sqrt(rmse2/len(valid_loader))

    acc = round(acc, 7)
    mean_f1 = round(mean_f1, 7)
    mIoU = round(mIoU, 7)
    mean_mae = round(mean_mae, 5)
    RMSE1 = round(RMSE1, 5)
    # RMSE2 = round(RMSE2, 5)

    
    y_OA.append(acc)
    y_F1.append(mean_f1)
    y_mIoU.append(mIoU)
    y_MAE.append(mean_mae)
    y_RMSE.append(RMSE1)
    y_cRMSE.append(RMSE2)
    print(f'Validation metrics - 2D: OA -> {acc*100} %; F1 Score -> {mean_f1*100} %; mIoU -> {mIoU*100} %; 3D: MAE -> {mean_mae} m; RMSE -> {RMSE1} m; cRMSE -> {RMSE2} m')

    if mean_f1 > best2dmetric:
      best2dmetric = mean_f1
      torch.save(net.state_dict(), out_file+'/2dbestnet.pth')
      print('Best 2D model saved!')

    if RMSE2 < best3dmetric:
      best3dmetric = RMSE2
      torch.save(net.state_dict(), out_file+'/3dbestnet.pth')  
      print('Best 3D model saved!')

  stats = dict(epoch = epoch, Loss2D = epoch_2d_loss, Loss3D = epoch_3d_loss, Loss = epoch_loss, RMSE = RMSE1, cRMSE = RMSE2, OA  = acc*100, F1Score = mean_f1*100, MIOU = mIoU*100)
#   print(json.dumps(stats), file=stats_file)
  print(json.dumps(stats, cls = MyEncoder), file=stats_file)

# stats2 = {'ListLoss2D':y_loss_2d, 'ListLoss3D': y_loss_3d, 'ListLoss':y_loss_total, 'ListOA': y_OA, 'ListF1':y_F1 , 'ListmIoU':y_mIoU , 'ListMAE':y_MAE , 'ListRMSE':y_RMSE, 'ListcRMSE':y_cRMSE}

stats2= dict(ListLoss2D = y_loss_2d, ListLoss3D= y_loss_3d, ListLoss= y_loss_total, ListOA= y_OA, ListF1=y_F1 , ListmIoU=y_mIoU , ListMAE=y_MAE , ListRMSE= y_RMSE)
# print(json.dumps(stats2), file=stats_file)
print(json.dumps(stats2, cls = MyEncoder), file=stats_file)

end = time.time()
print('Training completed. Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
print(f'Best metrics: F1 score -> {best2dmetric*100} %,\t cRMSE -> {best3dmetric}')

start = time.time()

if os.path.exists('%s/' % out_file + f'{res_cp}bestnet.pth'):
    net.load_state_dict(torch.load('%s/' % out_file + f'{res_cp}bestnet.pth'))
    print("Checkpoints correctly loaded: ", out_file)

net.eval()

TN = 0
FP = 0
FN = 0
TP = 0
mean_mae = 0
rmse1 = 0
rmse2 = 0
    
for t1, t2, mask2d, mask3d , name in tqdm(test_loader): 

  t1 = t1.to(device)
  t2 = t2.to(device)

  out2d, out3d = net(t1, t2)
  out2d = out2d.detach().argmax(dim=1)
  out2d = out2d.cpu().numpy()
  out3d = out3d.detach().cpu().numpy()
  out3d = (out3d + 1)*(max_scale - min_scale)/2 + min_scale #Tanh  

  try:
      tn, fp, fn, tp = metrics.confusion_matrix(mask2d.ravel(), out2d.ravel()).ravel()
  except: 
      tn, fp, fn, tp = [0,0,0,0]
      print('Only 0 mask') 

  mean_ae = metrics.mean_absolute_error(mask3d.ravel(), out3d.ravel())
  s_rmse1 = metric_mse(out3d.ravel(), mask3d.cpu().numpy().ravel(), mask2d.cpu().numpy().ravel(), exclude_zeros = False)
  s_rmse2 = metric_mse(out3d.ravel(), mask3d.cpu().numpy().ravel(), mask2d.cpu().numpy().ravel(), exclude_zeros = True)
  max_error = metrics.max_error(mask3d.ravel(), out3d.ravel())
  mask_max = np.abs(mask3d.cpu().numpy()).max()
  
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

  
mean_mae = mean_mae/len(test_loader)
acc = (TP+TN)/(TP+FP+TN+FN)  ####my
mean_f1 = 2*TP/(2*TP+FP+FN)
mIoU = TP/(TP+FN+FP)
RMSE1 = np.sqrt(rmse1/len(test_loader))
RMSE2 = np.sqrt(rmse2/len(test_loader))

acc = round(acc, 7)
mean_f1 = round(mean_f1, 7)
mIoU = round(mIoU, 7)
mean_mae = round(mean_mae, 5)
RMSE1 = round(RMSE1, 5)
# RMSE2 = round(RMSE2, 5)

end = time.time()
print('Test completed. Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
print(f'Test metrics - 2D: OA -> {acc*100} %; F1 Score -> {mean_f1*100} %; mIoU -> {mIoU*100} %; 3D: MAE -> {mean_mae} m; RMSE -> {RMSE1} m; cRMSE -> {RMSE2} m')
stats = dict(epoch = 'Test', MeanAbsoluteError = mean_mae, RMSE = RMSE1, cRMSE = RMSE2, OA  = acc*100, F1Score = mean_f1*100, MIOU = mIoU*100)
# print(json.dumps(stats), file=stats_file)
print(json.dumps(stats, cls = MyEncoder), file=stats_file)