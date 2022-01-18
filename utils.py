# Some utility functions

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch 
from torchvision.transforms.functional import normalize

# convert a discrete class map to a rgb image
def convert_class_to_rgb(img_class):
    class_color_map = {
        0:(0,0,0),
        1:(85,107,47), #ground
        2:(0,255,0), #vegetation
        3:(255,165,0), #building
        4:(41,49,101), #wall
        5:(0,0,0), #bridge
        6:(0,0,255), #parking
        7:(255,0,255), #rail
        8:(201,201,201), #traffic road
        9:(89,47,95), #street furniture
        10:(255,0,0), #car
        11:(255,255,0), #footpath
        12:(0,255,255), #bike
        13:(0,191,255), #water
    }
    h, w = img_class.shape
    img_rgb = np.zeros((h,w,3))
    for classlabel, color in class_color_map.items():
        binary_mask = (img_class == classlabel)
        img_rgb[binary_mask,:] = color 
    return img_rgb.astype(np.uint8)

# denormalize the image (useful for rgb images)
def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)


if __name__ == "__main__":

    '''
    # test the convert_class_to_rgb
    sem_img_path = "/mnt/NVMe-2TB/qiaojun/SensatUrban/birmingham_2/Semantics/0000.png"
    sem_img = Image.open(sem_img_path)
    sem_img = np.array(sem_img)
    sem_img_rgb = convert_class_to_rgb(sem_img)
    Image.fromarray(sem_img_rgb.astype(np.uint8)).save('img_predict.png')
    plt.subplot(121)
    plt.imshow(sem_img)
    plt.subplot(122)
    plt.imshow(sem_img_rgb)
    plt.show()
    '''