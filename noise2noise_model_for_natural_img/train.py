import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import os 
import torchvision.transforms.functional as tvF
from unet_model import UNet
from Config import Config as conf
import time
from data_set_builder import Training_Dataset
from torch.utils.data import Dataset, DataLoader

def save_model(model,epoch):
    '''save model for eval'''

    ckpt_name = '/denoise_epoch_{}.pth'.format(epoch)
    path = conf.data_path_checkpoint 
    if not  os.path.exists(path):
        os.mkdir(path)
    path_final = path + ckpt_name
    print('Saving checkpoint to: {}\n'.format(path_final))
    torch.save(model.state_dict(), path_final)


def train():
    device = torch.device(conf.cuda if torch.cuda.is_available() else "cpu")
    dataset = Training_Dataset(conf.data_path_train,conf.gaussian_noise_param,conf.crop_img_size)
    dataset_length = len(dataset)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True,num_workers=4)
    model = UNet(in_channels =conf.img_channel,out_channels=conf.img_channel)
    criterion = nn.MSELoss()
    model = model.to(device)
    optim = Adam(model.parameters(), lr = conf.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True)
    scheduler = lr_scheduler.StepLR(optim, step_size=100, gamma=0.5)
    model.train()
    print(model)
    print("Starting Training Loop...")
    since = time.time()
    for epoch in range(conf.max_epoch):
        print('Epoch {}/{}'.format(epoch, conf.max_epoch - 1))
        print('-' * 10)
        running_loss = 0.0
        scheduler.step()
        for batch_idx , (source,target) in enumerate(train_loader):

            source = source.to(device)
            target = target.to(device)
            optim.zero_grad()

            denoised_source = model(source)
            loss = criterion(denoised_source,target)
            loss.backward()
            optim.step()
            
            running_loss +=loss.item()*source.size(0)
            print('Current loss {} and current batch idx {}' .format(loss.item(),batch_idx))
        epoch_loss = running_loss /dataset_length
        print('{} Loss: {:.4f}'.format('current '+ str(epoch), epoch_loss))
        if (epoch + 1) % conf.save_per_epoch == 0:
            save_model(model,epoch + 1)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def main():


    train()


if(__name__=="__main__"):
    main()



