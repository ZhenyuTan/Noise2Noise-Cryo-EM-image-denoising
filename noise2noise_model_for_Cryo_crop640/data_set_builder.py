from __future__ import print_function, division
import os
import torch
from skimage import io
from skimage.transform import resize
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tvF


class TrainingDataset(Dataset):
    ## only need to clean the even data
    def __init__(self,even_image_dir,odd_image_dir,image_size):
        self.even_image_dir = even_image_dir
        self.odd_image_dir = odd_image_dir
        self.image_list = os.listdir(even_image_dir)
        self.image_size = image_size


    def __len__(self):
        return len(os.listdir(self.even_image_dir))

    def __getitem__(self,idx):
        image_even_path = os.path.join(self.even_image_dir,self.image_list[idx])
        img_even_in = io.imread(image_even_path)
        img_odd_path =  os.path.join(self.odd_image_dir,self.image_list[idx])
        img_odd_in = io.imread(img_odd_path)

        h, w = img_even_in.shape
        new_h, new_w = self.image_size,self.image_size
        top = np.random.randint(0,h-new_h+1)
        left = np.random.randint(0,w-new_w+1)
        cropped_even = img_even_in[top:top+new_h,left:left+new_w]
        cropped_odd =  img_odd_in[top:top+new_h,left:left+new_w]
        img_even = np.expand_dims(cropped_even, axis=-1)
        img_odd =  np.expand_dims(cropped_odd, axis=-1)
        source = tvF.to_tensor(img_even)
        target = tvF.to_tensor(img_odd)
        
        return source , target


class TestingDataset(Dataset):

    def __init__(self,image_dir,image_size):
        self.test_dir = image_dir
        self.test_list = os.listdir(image_dir)
        self.test_image_size = image_size


    def __len__(self):
        return len(os.listdir(self.test_dir))

    def __getitem__(self,idx):
        image_name = os.path.join(self.test_dir,self.test_list[idx])
        img = io.imread(image_name)
        input_temp = self.__crop_img(img)
        input_exdim = np.expand_dims(input_temp, axis=-1)
        img_input = tvF.to_tensor(input_exdim)
  
        return img_input


        
    def __crop_img(self,img):
        '''crop the img '''
        h, w = img.shape
        new_h, new_w = self.test_image_size,self.test_image_size
        top = np.random.randint(0,h-new_h+1)
        left = np.random.randint(0,w-new_w+1)
        cropped_img = img[top:top+new_h,left:left+new_w]

        return cropped_img

    




        
