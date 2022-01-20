import os 
from imageio import imread, imwrite
from scipy import ndimage
import torch 
import numpy as np

dataset_dir = "/mnt/NVMe-2TB/qiaojun/SensatUrban/"
dataset_name_list = ["birmingham_2", "birmingham_3", "birmingham_4", "birmingham_5", "birmingham_6", "cambridge_4",
                     "cambridge_5", "cambridge_6", "cambridge_10", "cambridge_11", "cambridge_12", "cambridge_14", "cambridge_15"]
sample_num_list = [500, 1000, 2000, 4000]
num_imgs = 660
depth_scale = 100

if __name__ == '__main__':
    for dataset_name in dataset_name_list:
        print(dataset_name)
        data_depth_dir = os.path.join(dataset_dir,dataset_name,"Depths")
        for idx in range(num_imgs):
            depth_img = imread(os.path.join(data_depth_dir,"%04d.png"%idx))
            depth_img[np.where(depth_img>65000)] = 0
            imwrite(os.path.join(data_depth_dir,"%04d.png"%idx), depth_img.astype(np.uint16))