# This script is used to generate the simplified label maps with only 5 classes
import os
import numpy as np
from PIL import Image
from semantic_labels import simplify_class_map_sensat

dataset_dir = "/mnt/NVMe-2TB/qiaojun/SensatUrban/"
dataset_name_list = ["birmingham_2","birmingham_3","birmingham_4","birmingham_5","birmingham_6","cambridge_4","cambridge_5","cambridge_6","cambridge_10","cambridge_11","cambridge_12","cambridge_14","cambridge_15"]

for dataset_name in dataset_name_list:
    curr_dir = os.path.join(dataset_dir,dataset_name)
    # First rename the original folder
    os.rename(os.path.join(curr_dir,"Semantics"),os.path.join(curr_dir,"Semantics_Full"))
    # Then transform each of the semantic map to a new one
    if not os.path.isdir(os.path.join(curr_dir,"Semantics_5")):
        os.mkdir(os.path.join(curr_dir,"Semantics_5"))
    for idx in range(660):
        sem_img_path_read = os.path.join(curr_dir,"Semantics_Full","%04d.png"%idx)
        sem_img_path_write = os.path.join(curr_dir,"Semantics_5","%04d.png"%idx)        
        sem_img = Image.open(sem_img_path_read)
        sem_img = np.array(sem_img)
        sem_img_new = simplify_class_map_sensat(sem_img)
        Image.fromarray(sem_img_new.astype(np.uint8)).save(sem_img_path_write)