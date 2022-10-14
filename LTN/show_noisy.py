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
from utils import mkdir,model_train,setup_seed
from loss import loss_WCE
from model import LTN
import os, json
# from network import WSDDN
from tensorboardX import SummaryWriter
#from loss import loss_ce
from easydict import EasyDict as edict
import time
from torchsummary import summary
from torch.optim import lr_scheduler
from loss import WCE
import cv2
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    setup_seed(20)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='config path')
    args = parser.parse_args()
    file_dir = args.config_path

    with open(file_dir) as f:
            config = json.load(f)

    config = edict(config)

    dataset = config['DATA']['dataset']
    img_size = config['DATA']['image_size']
    bs = config['TRAIN']['batch_size']
    lr = config['TRAIN']['lr']
    weight_decay = config['TRAIN']['weight_decay']
    lr_decay = config['TRAIN']['lr_decay']


    train_img = os.path.join(config['DATA']['data_dir'],config['DATA']['train_dir'])
    train_label = 'C:/Jue/Project/Label_noise_ACT/Un/result/act/20220822220542/train/y/'
    test_img = os.path.join(config['DATA']['data_dir'],config['DATA']['test_dir'])
    test_label = os.path.join(config['DATA']['data_dir'],config['DATA']['test_gt'])

    # train_data = dataLoader(img_path=train_img, label_path=train_label, sigma_path = train_sigma, augment_data=False, target_size=img_size)
    # val_data = dataLoader(img_path=test_img, label_path=test_label, sigma_path = test_sigma, augment_data=False, target_size=img_size)
    # test_data = InfDataloader(test_img, target_size=img_size)

    # train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)
    # val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=0)

    model_path = config.TEST.model_path

    model = LTN().to(device)
    model.load_state_dict(torch.load(model_path))

    image_list = os.listdir(train_img)
    save_dir = config.TEST.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for image_name in image_list:
        inp_img = cv2.imread(os.path.join(train_img, image_name)).astype('float32')
        inp_img /= 255.0
        inp_img = np.transpose(inp_img, axes=(2, 0, 1))
        inp_img = torch.from_numpy(inp_img).float().to(device)
        print(inp_img.shape)


        y = cv2.imread(os.path.join(train_label, image_name),0).astype('float32')
        _, mask = cv2.threshold(y,255/2.0,255,cv2.THRESH_BINARY)

        mask = torch.from_numpy(mask).float().to(device)

        mask = torch.unsqueeze(mask,2)
        mask = torch.cat(((1-mask),mask),2)
        mask = mask.unsqueeze(3)
        print(mask.shape)


        #sigma_img = np.load(os.path.join(train_sigma, image_name[:-3]+'npy'))
        #sigma_img =  torch.from_numpy(sigma_img).to(device)
        #sigma_img = torch.unsqueeze(sigma_img,0)
        #print(sigma_img.shape)

        T = model(inp_img.unsqueeze(0))
        print(T.shape)

        T = T.permute(0, 2, 3, 1)
        T1,T2,T3,T4 = T[:,:,:,0].unsqueeze(3),(1-T[:,:,:,1]).unsqueeze(3),(1-T[:,:,:,0]).unsqueeze(3),T[:,:,:,1].unsqueeze(3)
        Transition = torch.cat((T1,T2,T3,T4),3)
        
        #print(mask.shape)
        _,h,w,_ = Transition.shape
        Transition = Transition.reshape(h,w,2,2)
        print(Transition.shape)
        pre_T = torch.matmul(Transition,mask)
        pre_T = pre_T.squeeze(3)

        pre_T[pre_T<0] = 0
        pre_T[pre_T>1] = 1
        #print(pre_T)
        pre_T = pre_T.squeeze().cpu().detach().numpy()

        #np.save(os.path.join(save_dir_T, image_name[:-3]+'npy'),Transition)
        cv2.imwrite(os.path.join(save_dir, image_name), pre_T[:,:,1].squeeze()*255)
        #cv2.imwrite(os.path.join(save_dir_T, image_name[:-3]+'1.png'), T11*255)


        # break
