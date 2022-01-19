# Some utility functions related to semantic labels

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Convert a discrete class map in SensatUrban to a rgb image
# 14 classes
def convert_class_to_rgb_sensat_full(img_class):
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

# Convert a discrete class map in SensatUrban to a rgb image
# 5 classes
def convert_class_to_rgb_sensat_simplified(img_class):
    class_color_map = {
        0:(0,0,0),
        1:(85,107,47), #ground
        2:(0,255,0), #vegetation
        3:(255,165,0), #building
        4:(201,201,201), #traffic road
    }
    h, w = img_class.shape
    img_rgb = np.zeros((h,w,3))
    for classlabel, color in class_color_map.items():
        binary_mask = (img_class == classlabel)
        img_rgb[binary_mask,:] = color 
    return img_rgb.astype(np.uint8)

# Convert the class map in SensatUrban from 14 classes to 5 classes.
def simplify_class_map_sensat(img_class):
    class_map = {
        0: 0,
        1: 1, #ground
        2: 2, #vegetation
        3: 3, #building
        4: 3, #wall -> building
        5: 4, #bridge -> traffic road
        6: 4, #parking -> traffic road
        7: 1, #rail -> ground
        8: 4, #traffic road
        9: 1, #street furniture -> ground
        10: 4, #car -> traffic road
        11: 1, #footpath -> ground
        12: 1, #bike -> ground
        13: 0, #water -> unknown
    }
    h, w = img_class.shape
    img_class_new = np.zeros((h,w))
    for classlabel, color in class_map.items():
        binary_mask = (img_class == classlabel)
        img_class_new[binary_mask] = color 
    return img_class_new.astype(np.uint8)

if __name__ == "__main__":
    print("Hello World.")
