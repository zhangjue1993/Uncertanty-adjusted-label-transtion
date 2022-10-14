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
from loss import loss_WCE,WCE
from model import RAN,LTN
import os, json
# from network import WSDDN
from tensorboardX import SummaryWriter
#from loss import loss_ce
from easydict import EasyDict as edict
import time
from torchsummary import summary
from torch.optim import lr_scheduler
from loss import WCE, WCE_label_correction_noise,WCE_label_correction_un

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
    noise_rate = config['TRAIN']['noise_rate']


    train_img = os.path.join(config['DATA']['data_dir'],config['DATA']['train_dir'])
    train_label = os.path.join(config['DATA']['data_dir'],config['DATA']['cam_dir'])
    train_bayes_label = config['DATA']['bayes_optimal_label']
    train_sigma_dir = config['DATA']['train_sigma_dir']
    test_img = os.path.join(config['DATA']['data_dir'],config['DATA']['test_dir'])
    test_label = os.path.join(config['DATA']['data_dir'],config['DATA']['test_gt'])
    #test_sigma = config['DATA']['test_sigma_dir']


    train_data = dataLoader(img_path=train_img, label_path=train_label, optimal_label_path=train_bayes_label,sigma_path= train_sigma_dir, noise_rate = noise_rate, augment_data=False, target_size=img_size)
    val_data = InfDataloader(img_path=test_img, label_path=test_label, augment_data=False, target_size=img_size)
    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=0)


    time_now = time.strftime("%Y%m%d%H%M%S", time.localtime())
    model_path = os.path.join('./checkpoint/',dataset, time_now)
    log_dir = os.path.join('./log/',dataset, time_now)
    save_dir = os.path.join('./result/',dataset, time_now,'train')

    if not os.path.exists(model_path):
        mkdir(model_path)
    if not os.path.exists(log_dir):
        mkdir(log_dir)
    if not os.path.exists(save_dir):
        mkdir(save_dir)

    config.TEST.model_path = model_path
    config.TEST.save_dir = save_dir

    with open(file_dir, 'w') as f:
            f.write(json.dumps(config, indent=4))
    with open(os.path.join(log_dir,'config.json'),'w') as f:
        f.write(json.dumps(config, indent=4))

    model = LTN().to(device)
    if config['TRAIN']['Resume'] ==1:
        Y_path = config['TRAIN']['Path']

        pretrained = torch.load(Y_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k,v in pretrained.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    summary(model, (3, 256, 256))
    criterion = WCE_label_correction_un().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=lr_decay)

    writer = SummaryWriter(log_dir)
 

    model_train(model, train_dataloader, val_dataloader, criterion,optimizer, scheduler, writer, model_path, device, config)


