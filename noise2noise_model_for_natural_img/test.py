import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import os 
import numpy as np
import torchvision.transforms.functional as tvF
from unet_model import UNet
from Config import Config as conf
from skimage import io

from data_set_builder import Testinging_Dataset
from torch.utils.data import Dataset, DataLoader


def test():
    device = torch.device(conf.cuda if torch.cuda.is_available() else "cpu")
    test_dataset = Testinging_Dataset(conf.data_path_test,conf.test_noise_param,conf.crop_img_size)
    test_loader =  DataLoader(test_dataset, batch_size=1, shuffle=False)
    print('Loading model from: {}'.format(conf.model_path_test))
    model = UNet(in_channels=conf.img_channel,out_channels=conf.img_channel)
    print('loading model')
    model.load_state_dict(torch.load(conf.model_path_test))
    model.eval()
    model.to(device)
    result_dir = conf.denoised_dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    for batch_idx, (source,img_cropped) in enumerate(test_loader):
        source_img = tvF.to_pil_image(source.squeeze(0))
        img_truth = img_cropped.squeeze(0).numpy().astype(np.uint8)
        source = source.to(device)
        denoised_img = model(source).detach().cpu()
        
        img_name = test_loader.dataset.image_list[batch_idx]
        
        denoised_result= tvF.to_pil_image(torch.clamp(denoised_img.squeeze(0), 0, 1))
        fname = os.path.splitext(img_name)[0]
        
        source_img.save(os.path.join(result_dir, f'{fname}-noisy.png'))
        denoised_result.save(os.path.join(result_dir, f'{fname}-denoised.png'))       
        io.imsave(os.path.join(result_dir, f'{fname}-ground_truth.png'),img_truth)

def main():


    test()


if(__name__=="__main__"):
    main()