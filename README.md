# Noise2Noise-for-Cryo-EM-image-denoising
pytorch implementation of noise2noise for Cryo-EM image denoising 
https://arxiv.org/abs/1803.04189
# Network Architecture
Similar to the noise2noise paper 
![image](https://github.com/ZhenyuTan/Noise2Noise-for-Cryo-EM-image-denoising/blob/master/imgs/unet.png)
# Loss function 
L2 loss
# Dependencies
pytorch CUDA 9.0 CuDNN 7.0 Anaconda(python3.6)
# Training 
python train.py 
(you need to modify the path in the config.py)
# Testing 
python test.py
(you need to modify the path in the config.py)
# Results on the natrual imgs
![image](https://github.com/ZhenyuTan/Noise2Noise-for-Cryo-EM-image-denoising/blob/master/imgs/results_natrual.png)
train the network using 256x256-pixel crops drawn from the 5k images in the COCO 2017 validation set for 120 epoch.  We furthermore randomize the noise standard deviation σ= [0,50] separately for each training example.
# Results on Cryo-EM data
![image](https://github.com/ZhenyuTan/Noise2Noise-for-Cryo-EM-image-denoising/blob/master/imgs/results_cryo.png)

We train the network using 640*640 crops drawn from the 250 images for 500 epoch for each protein sample dataset. we tested on 2 protein sample dataset,one is aldolase, the other is apoferritin
