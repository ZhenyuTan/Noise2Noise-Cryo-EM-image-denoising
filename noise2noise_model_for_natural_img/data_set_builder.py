from __future__ import print_function, division
import os
import torch
from skimage import io
from skimage.transform import resize
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvF


class Training_Dataset(Dataset):

    def __init__(self,image_dir,noise_param,image_size,add_noise =True, crop = True):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)
        self.noise_param = noise_param
        self.image_size = image_size
        self.add_noise = add_noise
        self.crop_img = crop


    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self,idx):
        image_name = os.path.join(self.image_dir,self.image_list[idx])
        img = io.imread(image_name)
        img_cropped = self.__crop_img(img)
        source_temp = self.__add_noise(img_cropped)
        source = tvF.to_tensor(source_temp)
        target_temp = self.__add_noise(img_cropped)
        target = tvF.to_tensor(target_temp)           


        return source , target

    def __add_noise(self, img):
        '''add Gaussain noise'''
        h,w,c = img.shape

        std = np.random.uniform(0,self.noise_param)
        noise = np.random.normal(0,std,(h,w,c))
        noise_img_temp = img + noise
        noise_img = np.clip(noise_img_temp, 0, 255).astype(np.uint8)
        return noise_img
        
    def __crop_img(self,img):
        '''crop the img '''
        h, w,c = img.shape
        new_h, new_w = self.image_size,self.image_size
        if min(h,w) <  self.image_size:
            img = resize(img,(self.image_size,self.image_size),preserve_range=True)
        h_r,w_r,c = img.shape
        top = np.random.randint(0,h_r-new_h+1)
        left = np.random.randint(0,w_r-new_w+1)
        cropped_img = img[top:top+new_h,left:left+new_w]

        return cropped_img

class Testinging_Dataset(Dataset):

    def __init__(self,test_dir,noise_param,image_size,add_noise =True, crop = True):
        self.image_dir = test_dir
        self.image_list = os.listdir(test_dir)
        self.noise_param = noise_param
        self.image_size = image_size
        self.add_noise = add_noise
        self.crop_img = crop


    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self,idx):
        image_name = os.path.join(self.image_dir,self.image_list[idx])
        img = io.imread(image_name)
        img_cropped = self.__crop_img(img)
        source_temp = self.__add_noise(img_cropped)
        source = tvF.to_tensor(source_temp)        


        return source,img_cropped

    def __add_noise(self, img):
        '''add Gaussain noise'''
        h,w,c = img.shape

        std = self.noise_param
        noise = np.random.normal(0,std,(h,w,c))
        noise_img_temp = img + noise
        noise_img = np.clip(noise_img_temp, 0, 255).astype(np.uint8)
        return noise_img
        
    def __crop_img(self,img):
        '''crop the img '''
        h, w,c = img.shape
        new_h, new_w = self.image_size,self.image_size
        if min(h,w) <  self.image_size:
            img = resize(img,(self.image_size,self.image_size),preserve_range=True)
        h_r,w_r,c = img.shape
        top = np.random.randint(0,h_r-new_h+1)
        left = np.random.randint(0,w_r-new_w+1)
        cropped_img = img[top:top+new_h,left:left+new_w]

        return cropped_img

    




        
