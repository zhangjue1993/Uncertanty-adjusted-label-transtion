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
from dataloader1 import dataLoader, InfDataloader
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

            logistic, pred_masks = model(inp_imgs)
            loss = criterion(pred_masks, gt_masks.long())

            tot_loss += loss.item()

            

    avg_loss = tot_loss / batch_idx

    writer.add_scalar('Val/loss', avg_loss, ep)
   
    return avg_loss


def model_train(model, train_dataloader, val_dataloader,criterion, optimizer, scheduler, writer, model_path, device, config):
    k = 0
    best_test_mae = float('inf')
    epoch = config['TRAIN']['epoch_num']
    log_interval = config['TRAIN']['log_interval']
    val_interval = config['VAL']['val_interval']
    weight = config['TRAIN']['weight']

    for ep in range(epoch):

        model.train()

        #torch.autograd.set_detect_anomaly(True)

        #print(epoch)
        #print(train_dataloader)
        for batch_idx, (inp_imgs, gt_masks, beyes_mask,noisy_mask,sigma_img) in enumerate(train_dataloader):
            
            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)
            
            beyes_mask = beyes_mask.to(device)
            noisy_mask = noisy_mask.to(device)
            sigma_img =  sigma_img.to(device)

            #print(sigma_img.shape)
            # T = torch.reshape(T, (T.size()[0], 256, 256, 2, 2))
            # print(T.size())

            optimizer.zero_grad()
            logistic= model(inp_imgs)
            loss, noisy_out = criterion(beyes_mask, gt_masks,logistic,noisy_mask,sigma_img)
            loss =loss-weight*torch.mean(torch.sum(logistic,dim=1))
            loss.backward()

            optimizer.step()
            pre_T = noisy_out[1,:,:].squeeze().cpu().detach().numpy()
            cv2.imwrite(os.path.join(save_dir, image_name), pre_T[:,:,1].squeeze()*255)


            if batch_idx % log_interval == 0:
                k +=1
                print('TRAIN :: Epoch : {}\tBatch : {}/{} ({:.2f}%)\t\tTot Loss : {:.4f}'
                        .format(ep + 1, batch_idx + 1, len(train_dataloader), (batch_idx + 1) * 100 / len(train_dataloader),loss.item()))
                
                writer.add_scalar('Train/loss', loss.item(), k)
                writer.add_image('Train/input', torch.squeeze(inp_imgs[1,:,:,:]), k)
                writer.add_image('Train/output', noisy_out[1,:,:].unsqueeze(0), k)
                writer.add_image('Train/T00', logistic[1,0,:,:].unsqueeze(0), k)
                #writer.add_image('Train/T01', (1-logistic[1,0,:,:]).unsqueeze(0), k)
                #writer.add_image('Train/T10', (1-logistic[1,1,:,:]).unsqueeze(0), k)   
                writer.add_image('Train/T11', logistic[1,1,:,:].unsqueeze(0), k)

                writer.add_image('Train/pseudo_mask', gt_masks[1,:,:].unsqueeze(0), k)
                writer.add_image('Train/bayes_mask', beyes_mask[1,:,:].unsqueeze(0), k)
                writer.add_image('Train/noisy_mask', noisy_mask[1,:,:].unsqueeze(0), k)
                writer.add_image('Train/sigma_img', sigma_img[1,:,:].unsqueeze(0), k)



                writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], k)


        if optimizer.param_groups[0]['lr']>1e-4:
            scheduler.step()
            # Validation
        # if ep % val_interval == 0:
        #     model.eval()
        #     val_mae = model_val(model,val_dataloader,criterion1, ep, writer, device, config)

            # Save the best model
            # if val_mae < best_test_mae:
            #     best_test_mae = val_mae
            #     torch.save(model.state_dict(), os.path.join(model_path, 'best-model_epoch-{:03}_mae-{:.4f}.pth'.
            #                 format(ep, best_test_mae)))
        if ep % 5 == 0 and ep>1:
            torch.save(model.state_dict(), os.path.join(model_path, 'best-model_epoch-{:03}_mae-{:.4f}.pth'.
                            format(ep, best_test_mae)))


