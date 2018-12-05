class Config:
    training_even = '/lsi/groups/mcianfroccolab/zhenyutan/noise2noise_test/data/apofretin_data/training_set/even'
    training_odd = '/lsi/groups/mcianfroccolab/zhenyutan/noise2noise_test/data/apofretin_data/training_set/odd'
    data_path_test ='/lsi/groups/mcianfroccolab/zhenyutan/noise2noise_test/data/apofretin_data/test_set'
    data_path_checkpoint ='/lsi/groups/mcianfroccolab/zhenyutan/noise2noise_test/data/apofretin_data/checkpoints_size640_500epo'
    model_path_test='/lsi/groups/mcianfroccolab/zhenyutan/noise2noise_test/data/apofretin_data/checkpoints_size640_500epo/denoise_epoch_500.pth'
    denoised_dir = '/lsi/groups/mcianfroccolab/zhenyutan/noise2noise_test/data/apofretin_data/denoised_size640_500epo'
    img_channel = 1
    max_epoch = 500
    crop_img_size = 640
    learning_rate = 0.001
    save_per_epoch = 20
    cuda = "cuda:0"

