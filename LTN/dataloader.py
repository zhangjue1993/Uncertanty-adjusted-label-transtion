from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import sys
import cv2
import numpy as np
import glob
import os,json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tr
import math
import copy


class dataLoader(Dataset):
    """
    DataLoader for training.
    """
    def __init__(self, img_path, label_path, optimal_label_path, sigma_path,noise_rate=0.3, augment_data=False, target_size=256):
        if os.path.exists(img_path) and os.path.exists(label_path) and os.path.exists(sigma_path) and os.path.exists(optimal_label_path):
            self.inp_path = img_path
            self.out_path = label_path
            self.sigma_path = sigma_path
            self.bayes_path = optimal_label_path
            self.noise_rate = noise_rate
        else:
            print(img_path,label_path,sigma_path,optimal_label_path)
            print("Please check the input and output path!")
            sys.exit(0)
        self.augment_data = augment_data
        self.target_size = target_size
        self.inp_files = os.listdir(self.inp_path)


    def __getitem__(self, idx):
        
        img_path = os.path.join(self.inp_path, self.inp_files[idx][:-3]+'png')
        label_path = os.path.join(self.out_path, self.inp_files[idx][:-3]+'png')
        bayes_label_path = os.path.join(self.bayes_path, self.inp_files[idx][:-3]+'png')
        sigma_path = os.path.join(self.sigma_path, self.inp_files[idx][:-3]+'png')
        if os.path.exists(sigma_path):

            inp_img = cv2.imread(img_path)
            mask_img = cv2.imread(label_path, 0)
            bayes_mask_img = cv2.imread(bayes_label_path, 0)
            sigma_img = cv2.imread(sigma_path, 0)
            #print('sigma:',sigma_img.shape)

            noisy_mask = np.where(np.abs(bayes_mask_img/255.0-0.5)>(self.noise_rate/2),1,0)
            _, mask_img = cv2.threshold(mask_img,255/2.0,255,cv2.THRESH_BINARY)
            _, bayes_mask_img = cv2.threshold(bayes_mask_img,255/2.0,255,cv2.THRESH_BINARY)
        

            mask_img = mask_img.astype('float32')
            bayes_mask_img = bayes_mask_img.astype('float32')
            inp_img = inp_img.astype('float32')
            sigma_img = sigma_img.astype('float32')


            inp_img /= 255.0
            inp_img = np.transpose(inp_img, axes=(2, 0, 1))
            mask_img /= 255.0
            bayes_mask_img /= 255.0
            sigma_img /=255.0

        else:
            print("Please check the images and labels!")
            print(img_path)
            sys.exit(0)

        return torch.from_numpy(inp_img).float(), torch.from_numpy(mask_img).float(),torch.from_numpy(bayes_mask_img).float(),torch.from_numpy(noisy_mask).float(), torch.from_numpy(sigma_img).float()

    def __len__(self):
        return len(self.inp_files)

class InfDataloader(Dataset):
    """
    DataLoader for training.
    """
    def __init__(self, img_path, label_path, augment_data=False, target_size=256):
        if os.path.exists(img_path) and os.path.exists(label_path):
            self.inp_path = img_path
            self.out_path = label_path
        else:
            print(img_path,label_path)
            print("Please check the input and output path!")
            sys.exit(0)
        self.augment_data = augment_data
        self.target_size = target_size
        self.inp_files = os.listdir(self.inp_path)


    def __getitem__(self, idx):
        
        img_path = os.path.join(self.inp_path, self.inp_files[idx])
        label_path = os.path.join(self.out_path, self.inp_files[idx])

        if os.path.exists(img_path) and os.path.exists(label_path):

            inp_img = cv2.imread(img_path)
            mask_img = cv2.imread(label_path, 0)
            _, mask_img = cv2.threshold(mask_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask_img = mask_img.astype('float32')
            inp_img = inp_img.astype('float32')

            inp_img /= 255.0
            inp_img = np.transpose(inp_img, axes=(2, 0, 1))
            mask_img /= 255.0

        else:
            print("Please check the images and labels!")
            print(img_path)
            sys.exit(0)

        return torch.from_numpy(inp_img).float(), torch.from_numpy(mask_img).long()

    def __len__(self):
        return len(self.inp_files)




# class InfDataloader(Dataset):
#     """
#     Dataloader for Inference.
#     """
#     def __init__(self, img_folder, target_size=256):
#         self.imgs_folder = img_folder
#         self.img_paths = sorted(glob.glob(self.imgs_folder + '/*'))

#         self.target_size = target_size
#         # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#         #                                       std=[0.229, 0.224, 0.225])

#     def __getitem__(self, idx):
#         """
#         __getitem__ for inference
#         :param idx: Index of the image
#         :return: img_np is a numpy RGB-image of shape H x W x C with pixel values in range 0-255.
#         And img_tor is a torch tensor, RGB, C x H x W in shape and normalized.
#         """
#         img = cv2.imread(self.img_paths[idx])
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Pad images to target size
#         img_np = resize_image(img, self.target_size)
#         img_tor = img_np.astype(np.float32)
#         img_tor = img_tor / 255.0
#         img_tor = np.transpose(img_tor, axes=(2, 0, 1))
#         img_tor = torch.from_numpy(img_tor).float()
#         #img_tor = self.normalize(img_tor)

#         return img_np, img_tor

#     def __len__(self):
#         return len(self.img_paths)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='config path')
    args = parser.parse_args()
    file_dir = args.config_path

    with open(file_dir) as f:
            config = json.load(f)
    print(config)
    img_size = config['DATA']['image_size']
    bs = config['TRAIN']['batch_size']
    train_img = os.path.join(config['DATA']['data_dir'],config['DATA']['train_dir'])
    train_label = os.path.join(config['DATA']['data_dir'],config['DATA']['cam_dir'])
    test_img = os.path.join(config['DATA']['data_dir'],config['DATA']['test_dir'])
    test_label = os.path.join(config['DATA']['data_dir'],config['DATA']['test_gt'])
    train_data = dataLoader(img_path=train_img, label_path=train_label, augment_data=False, target_size=img_size)
    val_data = dataLoader(img_path=test_img, label_path=test_label, augment_data=False, target_size=img_size)
    test_data = InfDataloader(test_img, target_size=img_size)

    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=2)

    print("Train Dataloader :")
    for batch_idx, (inp_imgs, gt_masks) in enumerate(train_dataloader):
        print('Loop :', batch_idx, inp_imgs.size(), gt_masks.size())
        if batch_idx == 3:
            break

    print("\nTest Dataloader :")
    for batch_idx, (inp_imgs, gt_masks) in enumerate(test_dataloader):
        print('Loop :', batch_idx, inp_imgs.size(), gt_masks.size())
        if batch_idx == 3:
            break

    # # Test image augmentation functions
    # inp_img = cv2.imread('./data/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00000003.jpg')
    # out_img = cv2.imread('./data/DUTS/DUTS-TE/DUTS-TE-Mask/ILSVRC2012_test_00000003.png', -1)
    # # inp_img = inp_img.astype('float32')
    # out_img = out_img.astype('float32')
    # out_img = out_img / 255.0

    # cv2.imshow('Original Input Image', inp_img)
    # cv2.imshow('Original Output Image', out_img)

    # print('\nImage shapes before processing :', inp_img.shape, out_img.shape)
    # x, y = random_crop_flip(inp_img, out_img)
    # x, y = random_rotate(x, y)
    # x = random_brightness(x)
    # x, y = resize_image(x, target_size=256), resize_image(y, target_size=256)
    # # x now contains float values, so either round-off the values or convert the pixel range to 0-1.
    # x = x / 255.0
    # print('Image shapes after processing :', x.shape, y.shape)

    # cv2.imshow('Processed Input Image', x)
    # cv2.imshow('Processed Output Image', y)
    # cv2.waitKey(0)