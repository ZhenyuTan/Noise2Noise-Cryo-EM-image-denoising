class Config:
    data_path_train = '/lsi/groups/mcianfroccolab/zhenyutan/noise2noise_test/data/noise2noise_COCO_data/training_set'
    data_path_test = '/lsi/groups/mcianfroccolab/zhenyutan/noise2noise_test/data/noise2noise_COCO_data/test_set'
    data_path_checkpoint ='/lsi/groups/mcianfroccolab/zhenyutan/noise2noise_test/data/noise2noise_COCO_data/checkpoints_120epo_sigma50'
    model_path_test='/lsi/groups/mcianfroccolab/zhenyutan/noise2noise_test/data/noise2noise_COCO_data/checkpoints_120epo_sigma50/denoise_epoch_120.pth'
    denoised_dir =  '/lsi/groups/mcianfroccolab/zhenyutan/noise2noise_test/data/noise2noise_COCO_data/test_results_120epo_sigma70'
    img_channel = 3
    max_epoch = 200
    crop_img_size = 256
    learning_rate = 0.001
    save_per_epoch = 20
    gaussian_noise_param = 30
    test_noise_param = 70
    cuda = "cuda:1"
