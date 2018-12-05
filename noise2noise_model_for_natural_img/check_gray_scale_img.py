import os 
from skimage import io
image_dir = '/lsi/groups/mcianfroccolab/zhenyutan/noise2noise_test/data/noise2noise_COCO_data/training_set'

for item in os.listdir(image_dir):
    img_path = os.path.join(image_dir,item)
    img = io.imread(img_path)
    shape = img.shape
    if len(shape)==2:
        print(item)
