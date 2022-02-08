# Use a pretrained 2D segmentation model to derive the segmentation mask
import os
import numpy as np
from imageio import imread

import torch
import torch.nn as nn

from config import get_sensat_cfg
from model.deeplab import deeplabv3_resnet18, deeplabv3_resnet34, deeplabv3_resnet50

num_imgs = 660
cfg_file = "Sensat_deeplab.yaml"
save_model_path = "checkpoints_others/deeplab/0125_1617_deeplab_resnet18_train_mesh1024_depth1000_channel3_focal_loss_50_0.01/model_49.tar"
#save_model_path = "checkpoints_others/deeplab/0125_1619_deeplab_resnet34_train_mesh1024_depth1000_channel3_focal_loss_50_0.01/model_best_semantic.tar"
#save_model_path = "checkpoints_others/deeplab/0124_2331_train_mesh1024_depth1000_channel3_focal_loss_50_0.01/model_best_semantic.tar"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

dataset_dir = "/mnt/NVMe-2TB/qiaojun/SensatUrban/"
dataset_name_list = ["birmingham_2", "birmingham_3", "birmingham_4", "birmingham_5", "birmingham_6", "cambridge_4",
                     "cambridge_5", "cambridge_6", "cambridge_10", "cambridge_11", "cambridge_12", "cambridge_14", "cambridge_15"]

cfg = get_sensat_cfg()
cfg.merge_from_file(cfg_file)

model = deeplabv3_resnet18(cfg)
#model = deeplabv3_resnet34(cfg)
checkpoint = torch.load(save_model_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)

softmax_fn = nn.Softmax(dim=1)

for dataset_name in dataset_name_list:
    print(dataset_name)
    curr_dir = os.path.join(dataset_dir, dataset_name)

    if not os.path.isdir(os.path.join(curr_dir, "Semantics_2D")):
        os.mkdir(os.path.join(curr_dir, "Semantics_2D"))

    for idx in range(num_imgs):
        if (idx%10==0):
            print(idx)
        input_img = torch.Tensor(imread(os.path.join(curr_dir,"Images","%04d.png" % idx)))
        input_img = input_img.permute(2,0,1).unsqueeze(0).to(device, dtype=torch.float32)
        pred_semantic = model(input_img)
        pred_semantic = pred_semantic[0,::].detach().cpu()
        #pred_semantic = softmax_fn(pred_semantic/100)[0,::].detach().cpu()
        torch.save(pred_semantic, os.path.join(curr_dir, "Semantics_2D", "%04d.pt" % idx))
        