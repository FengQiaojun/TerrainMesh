import os 
from imageio import imread, imwrite
from scipy import ndimage
import torch 
import matplotlib.pyplot as plt
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
        for samples in sample_num_list:
            print(samples)
            data_depth_sparse_dir = os.path.join(dataset_dir,dataset_name,"Pcds_"+str(samples))
            for idx in range(num_imgs):
                sparse_depth_img = imread(os.path.join(data_depth_sparse_dir,"%04d.png"%idx))/depth_scale
                sparse_depth_mask = (sparse_depth_img<=0)*1
                depth_edt = ndimage.distance_transform_edt(sparse_depth_mask)
                torch.save(torch.from_numpy(depth_edt),os.path.join(data_depth_sparse_dir,"%04d_edt.pt"%idx))