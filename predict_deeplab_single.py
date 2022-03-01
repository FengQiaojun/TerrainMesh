# Load a PyTorch deeplab and train

import numpy as np
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from imageio import imread,imwrite

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from config import get_sensat_cfg
from dataset.build_data_loader import build_data_loader
from loss import FocalLoss
from model.deeplab import deeplabv3_resnet18, deeplabv3_resnet34, deeplabv3_resnet50
from utils.model_record_name import generate_segmodel_record_name
from utils.semantic_labels import convert_class_to_rgb_sensat_simplified
from utils.stream_metrics import StreamSegMetrics

cfg_file = "Sensat_deeplab.yaml"
image_list = ["/mnt/NVMe-2TB/qiaojun/journal_terrain/visualizations/0231.png"]

if __name__ == "__main__":
    # Load the config and create a folder to save the outputs.
    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)

    
    # Specify the GPU. Here use 2 of them.
    worker_id = cfg.SOLVER.GPU_ID
    device = torch.device("cuda:%d" % worker_id if torch.cuda.is_available() else 'cpu')
    
    # Build the model

    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    model.classifier[4] = nn.Conv2d(256, cfg.MODEL.DEEPLAB.NUM_CLASSES, kernel_size=1, stride=1)
    model.to(device)
    
    
    if cfg.MODEL.RESUME:
        save_model_path = cfg.MODEL.RESUME_MODEL
        save_path = os.path.join(cfg.MODEL.RESUME_MODEL,"..")
        #cfg.merge_from_file(os.path.join(save_path,"Sensat_basic.yaml"))
        checkpoint = torch.load(save_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()


    with torch.no_grad():

        for image_name in image_list:
            rgb_img = np.asfarray(imread(image_name)/255, dtype=np.float32)
            # Concatenate the inputs

            transform = [transforms.ToTensor()]
            # do imagenet normalization
            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]
            transform.append(transforms.Normalize(
                    mean=IMAGENET_MEAN, std=IMAGENET_STD))
            transform = transforms.Compose(transform)
            input_img = rgb_img
            input_img = transform(input_img)
            input_img = torch.unsqueeze(input_img, 0)
            input_img = input_img.to(device, dtype=torch.float32)
            if cfg.MODEL.BACKBONE == "resnet50":
                pred_semantic = model(input_img)["out"]                
            else:
                pred_semantic = model(input_img)             

            preds = pred_semantic.detach().max(dim=1)[1].cpu().numpy()
            
            
            plt.subplot(121)
            plt.imshow(rgb_img)
            plt.subplot(122)
            vis_preds = convert_class_to_rgb_sensat_simplified(preds[0,::])
            plt.imshow(vis_preds)
            plt.show()
            
            imwrite(image_name[:-4]+"_seg.png",vis_preds)