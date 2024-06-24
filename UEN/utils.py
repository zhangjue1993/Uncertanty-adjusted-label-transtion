from numpy.lib.twodim_base import mask_indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
#from torchsummary import summary
from torch.autograd import Variable
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from dataloader import dataLoader, InfDataloader
from utils import *
import os, json
import numpy as np
# from network import WSDDN
from tensorboardX import SummaryWriter
#from loss import loss_ce
from easydict import EasyDict as edict
import time
from torchsummary import summary
import cv2
from torch.optim import lr_scheduler
import random

from torch.autograd import Variable
from grid_sample import grid_sample
import itertools


def toLabel(input):
    temp = 1-input 
    newlabel = torch.cat((input, temp), 0)
    return newlabel


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        return False

def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def model_val(model,val_dataloader,criterion, ep, writer, device, config):
    
    model.eval()
    img_size = config['DATA']['image_size']
    tot_loss = 0
    tp_tf = 0   # TruePositive + TrueNegative, for accuracy
    tp = 0      # TruePositive
    pred_true = 0   # Number of '1' predictions, for precision
    gt_true = 0     # Number of '1's in gt mask, for recall
    mae_list = []   # List to save mean absolute error of each image

    with torch.no_grad():
        for batch_idx, (inp_imgs, gt_masks) in enumerate(val_dataloader, start=1):
            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)

            _,_,pred_masks = model(inp_imgs)
            loss = criterion(pred_masks, gt_masks.long())

            tot_loss += loss.item()

        

    writer.add_scalar('Val/loss', avg_loss, ep)
 
    writer.add_image('val/input', inp_imgs[5,:,:,:].squeeze(0), ep)
    writer.add_image('val/output', pred_masks[5,1,:,:].unsqueeze(0), ep)
    writer.add_image('val/gt', gt_masks[5,:,:].unsqueeze(0), ep)



    return avg_loss



def model_train_com(model, train_dataloader, val_dataloader, criterion1, criterion2,criterion3,optimizer, scheduler, writer, model_path, device, config,CRFconfig):
    k = 0
    best_test_mae = float('inf')
    epoch = config['TRAIN']['epoch_num']
    log_interval = config['TRAIN']['log_interval']
    val_interval = config['VAL']['val_interval']
    crf_w = config['TRAIN']['crf_w']
    wce = config['TRAIN']['wce']
    sigma_weight = config['TRAIN']['sigma_weight']

    for ep in range(epoch):

        model.train()

        #torch.autograd.set_detect_anomaly(True)

        #print(epoch)
        for batch_idx, (inp_imgs, gt_masks) in enumerate(train_dataloader):

            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)
            #print('input_img:',inp_imgs.shape)
            optimizer.zero_grad()
            pred_masks=[]
            logvar = []
            for i in range(3):
                _, logva,pred_mask= model(inp_imgs)
                pred_masks.append(pred_mask)
                logvar.append(logva)

            pred_masks = torch.stack(pred_masks,0)
            logvar =  torch.stack(logvar,0)
            #inp = inp_imgs.clone()
            #trans_inp =  trans(inp, inp.shape[3], rand_seed)
            #trans_pred = model(trans_inp)
            
            #trans_mask = trans(pred_masks[:,1,:,:].unsqueeze(1), pred_masks.shape[3], rand_seed)
            temp = pred_masks
            temp_var = logvar
            var_y = torch.mean(temp**2,dim=0)-torch.mean(temp,dim=0)**2+torch.mean(temp_var**2,dim=0)

            mean_y = torch.mean(pred_masks,dim=0)
            #print('mean_y:',mean_y.shape)
            #print('var_y:',var_y.shape)

            #print(pred_masks.size())
            loss1 = criterion1(mean_y, gt_masks)
            #loss2 = criterion2(mean_y[:,1,:,:].unsqueeze(1), CRFconfig,inp_imgs)
            #loss2 = criterion2(mean_y, CRFconfig,inp_imgs)
            #loss3 = criterion4(trans_pred[:,1,:,:].unsqueeze(1), trans_mask)
            loss = loss1+sigma_weight*torch.mean(var_y**2)
            

            #with torch.autograd.detect_anomaly():
            loss.backward()

            optimizer.step()

            if batch_idx % log_interval == 0:
                k +=1
                print('TRAIN :: Epoch : {}\tBatch : {}/{} ({:.2f}%)\t\tTot Loss : {:.4f}'
                        .format(ep + 1, batch_idx + 1, len(train_dataloader), (batch_idx + 1) * 100 / len(train_dataloader),loss.item()))
                writer.add_scalar('Train/loss', loss.item(), k)



        if optimizer.param_groups[0]['lr']>1e-5:
            scheduler.step()
            # Validation
        if ep % val_interval == 0:
            model.eval()
            val_mae = model_val(model,val_dataloader,criterion3, ep, writer, device, config)

            # Save the best model
            # if val_mae < best_test_mae:
            #     best_test_mae = val_mae
            #     torch.save(model.state_dict(), os.path.join(model_path, 'best-model_epoch-{:03}_mae-{:.4f}.pth'.
            #                 format(ep, best_test_mae)))
        if ep % 2 == 0 and ep>1:
            torch.save(model.state_dict(), os.path.join(model_path, 'best-model_epoch-{:03}_mae-{:.4f}.pth'.
                            format(ep, best_test_mae)))

