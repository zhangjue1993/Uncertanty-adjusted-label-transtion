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

            mask_img = pred_masks.cpu().numpy()
            mask = mask_img[:,1,:,:]
            mask[mask>=0.5] = 1
            mask[mask<0.5] = 0
            gts = gt_masks[:,:,:].cpu().numpy()
            
            for  i in range(len(inp_imgs)):
                gt =  gts[i,:,:]
                mask.astype(np.float)
                gt.astype(np.float)
                tp_tf += np.sum(mask == gt)
                tp += np.multiply(mask, gt).sum()
                pred_true += np.sum(mask)
                gt_true += np.sum(gt)
            

    avg_loss = tot_loss / batch_idx
    accuracy = tp_tf / (len(val_dataloader) *img_size * img_size)
    precision = tp / (pred_true + 1e-5)
    recall = tp / (gt_true + 1e-5)
    F = 2./(1./precision + 1./recall)

    writer.add_scalar('Val/loss', avg_loss, ep)
    writer.add_scalar('Val/accuracy', accuracy, ep)
    writer.add_scalar('Val/precision', precision, ep)
    writer.add_scalar('Val/recall', recall, ep)
    writer.add_scalar('Val/F', F, ep)

    
    writer.add_image('val/input', inp_imgs[5,:,:,:].squeeze(0), ep)
    writer.add_image('val/output', pred_masks[5,1,:,:].unsqueeze(0), ep)
    writer.add_image('val/gt', gt_masks[5,:,:].unsqueeze(0), ep)


    print('val :: ACC : {:.4f}\tPRE : {:.4f}\tREC : {:.4f}\tF : {:.4f}\tAVG-LOSS : {:.4f}\n'.format(
                                                                                            accuracy,
                                                                                            precision,
                                                                                            recall,
                                                                                            F,
                                                                                            avg_loss))

    return avg_loss


def model_train(model, train_dataloader, val_dataloader, criterion1, criterion2,criterion3,optimizer, scheduler, writer, model_path, device, config,CRFconfig):
    k = 0
    best_test_mae = float('inf')
    epoch = config['TRAIN']['epoch_num']
    log_interval = config['TRAIN']['log_interval']
    val_interval = config['VAL']['val_interval']
    crf_w = config['TRAIN']['crf_w']
    wce = config['TRAIN']['wce']


    for ep in range(epoch):

        model.train()

        #torch.autograd.set_detect_anomaly(True)

        #print(epoch)
        for batch_idx, (inp_imgs, gt_masks) in enumerate(train_dataloader):

            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)
            #print('input_img:',inp_imgs.shape)
            optimizer.zero_grad()
            pred_masks= model(inp_imgs)

            #inp = inp_imgs.clone()
            #trans_inp =  trans(inp, inp.shape[3], rand_seed)
            #trans_pred = model(trans_inp)
            
            #trans_mask = trans(pred_masks[:,1,:,:].unsqueeze(1), pred_masks.shape[3], rand_seed)
            

            #print(pred_masks.size())
            loss1 = criterion1(pred_masks, gt_masks)
            #loss2 = 0#criterion2(pred_masks[:,1,:,:].unsqueeze(1), CRFconfig,inp_imgs)
            #loss2 = criterion2(pred_masks, CRFconfig,inp_imgs)
            #loss3 = criterion4(trans_pred[:,1,:,:].unsqueeze(1), trans_mask)
            loss = loss1#+crf_w*loss2
            

            #with torch.autograd.detect_anomaly():
            loss.backward()

            optimizer.step()

            if batch_idx % log_interval == 0:
                k +=1
                print('TRAIN :: Epoch : {}\tBatch : {}/{} ({:.2f}%)\t\tTot Loss : {:.4f}'
                        .format(ep + 1, batch_idx + 1, len(train_dataloader), (batch_idx + 1) * 100 / len(train_dataloader),loss.item()))
                writer.add_scalar('Train/loss', loss.item(), k)
                #writer.add_scalar('Train/celoss', loss1.item(), k)
                #writer.add_scalar('Train/crfloss', loss2, k)
                #writer.add_scalar('Train/tfsloss', loss3.item(), k)
                writer.add_image('Train/input', torch.squeeze(inp_imgs[1,:,:,:]), k)
                #writer.add_image('Train/logvar', logvar[1,0,:,:].unsqueeze(0), k)
                #_image('Train/trans_inp', torch.squeeze(trans_inp[1,:,:,:]), k)
                writer.add_image('Train/output', pred_masks[1,1,:,:].unsqueeze(0), k)
                # writer.add_image('Train/trans_out', trans_mask[1,0,:,:].unsqueeze(0), k)
                # writer.add_image('Train/trans_pred', trans_pred[1,1,:,:].unsqueeze(0), k)

                writer.add_image('Train/gt', gt_masks[1,:,:].unsqueeze(0), k)
                writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], k)


        if ep>4 and optimizer.param_groups[0]['lr']>1e-5:
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
        if ep % 5 == 0 and ep>1:
            torch.save(model.state_dict(), os.path.join(model_path, 'best-model_epoch-{:03}_mae-{:.4f}.pth'.
                            format(ep, best_test_mae)))


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
                writer.add_scalar('Train/celoss', loss1.item(), k)
                #writer.add_scalar('Train/crfloss', loss2, k)
                #writer.add_scalar('Train/tfsloss', loss3.item(), k)
                
                writer.add_image('Train/logvar', var_y[1,1,:,:].unsqueeze(0)*3, k)
                #_image('Train/trans_inp', torch.squeeze(trans_inp[1,:,:,:]), k)
                writer.add_image('Train/output', mean_y[1,1,:,:].unsqueeze(0), k)
                #writer.add_image('Train/output_out', pred_out[1,1,:,:].unsqueeze(0), k)
                # writer.add_image('Train/trans_out', trans_mask[1,0,:,:].unsqueeze(0), k)
                # writer.add_image('Train/trans_pred', trans_pred[1,1,:,:].unsqueeze(0), k)
                writer.add_image('Train/input', torch.squeeze(inp_imgs[1,:,:,:]), k)
                writer.add_image('Train/gt', gt_masks[1,:,:].unsqueeze(0), k)
                writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], k)


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


def model_train_un(model, train_dataloader, val_dataloader, criterion1, criterion2,criterion3,optimizer, scheduler, writer, model_path, device, config,CRFconfig):
    k = 0
    best_test_mae = float('inf')
    epoch = config['TRAIN']['epoch_num']
    log_interval = config['TRAIN']['log_interval']
    val_interval = config['VAL']['val_interval']
    crf_w = config['TRAIN']['crf_w']
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
            pred_masks,logvar= model(inp_imgs)

           
            loss1 = criterion1(pred_masks, gt_masks)
            #loss2 = 0#criterion2(pred_masks[:,1,:,:].unsqueeze(1), CRFconfig,inp_imgs)
            #loss2 = criterion2(pred_masks, CRFconfig,inp_imgs)
            #loss3 = criterion4(trans_pred[:,1,:,:].unsqueeze(1), trans_mask)
            loss = loss1+sigma_weight*torch.mean(logvar**2)
            

            #with torch.autograd.detect_anomaly():
            loss.backward()

            optimizer.step()

            if batch_idx % log_interval == 0:
                k +=1
                print('TRAIN :: Epoch : {}\tBatch : {}/{} ({:.2f}%)\t\tTot Loss : {:.4f}'
                        .format(ep + 1, batch_idx + 1, len(train_dataloader), (batch_idx + 1) * 100 / len(train_dataloader),loss.item()))
                writer.add_scalar('Train/loss', loss.item(), k)
                #writer.add_scalar('Train/celoss', loss1.item(), k)
                #writer.add_scalar('Train/crfloss', loss2, k)
                #writer.add_scalar('Train/tfsloss', loss3.item(), k)
                writer.add_image('Train/input', torch.squeeze(inp_imgs[1,:,:,:]), k)
                writer.add_image('Train/output', pred_masks[1,1,:,:].unsqueeze(0), k)
                writer.add_image('Train/logvar', logvar[1,1,:,:].unsqueeze(0)*4, k)
                writer.add_image('Train/gt', gt_masks[1,:,:].unsqueeze(0), k)
                #_image('Train/trans_inp', torch.squeeze(trans_inp[1,:,:,:]), k)
                
                # writer.add_image('Train/trans_out', trans_mask[1,0,:,:].unsqueeze(0), k)
                # writer.add_image('Train/trans_pred', trans_pred[1,1,:,:].unsqueeze(0), k)

                
                writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], k)


        if optimizer.param_groups[0]['lr']>1e-4:
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
        if ep % 5 == 0 and ep>1:
            torch.save(model.state_dict(), os.path.join(model_path, 'best-model_epoch-{:03}_mae-{:.4f}.pth'.
                            format(ep, best_test_mae)))


