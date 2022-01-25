# Count the label distribution of the dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from config import get_sensat_cfg
from dataset.build_data_loader import build_data_loader

cfg_file = "Sensat_basic.yaml"

if __name__ == "__main__":
    # Load the config and create a folder to save the outputs.
    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build the DataLoaders 
    loaders = {}
    loaders["train"] = build_data_loader(cfg, "SensatSemantic", split_name=cfg.DATASETS.TRAINSET, num_workers=cfg.DATASETS.NUM_THREADS)
    loaders["val"] = build_data_loader(cfg, "SensatSemantic", split_name=cfg.DATASETS.VALSET, num_workers=cfg.DATASETS.NUM_THREADS)
    batch_num_train = int(np.ceil(len(loaders["train"].dataset)/loaders["train"].batch_size))
    batch_num_val = int(np.ceil(len(loaders["val"].dataset)/loaders["val"].batch_size))
    print("Training set size %d. Training batch number %d."%(len(loaders["train"].dataset),batch_num_train))
    print("Validation set size %d. Validation batch number %d."%(len(loaders["val"].dataset),batch_num_val))
    
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 5, kernel_size=1, stride=1)
    model.backbone.conv1 = nn.Conv2d(cfg.MODEL.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
    checkpoint = torch.load("checkpoints/0123_2242_train_mesh1024_depth1000_channel4_300_0.01/model_100.tar")
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    count_pixels = torch.zeros(5,1)
    #loop = tqdm(enumerate(loaders["train"]),total=batch_num_train)
    loop = tqdm(enumerate(loaders["val"]),total=batch_num_val)
    for i, batch in loop:
        rgb_img, sparse_depth, depth_edt, init_mesh_render_depth, gt_semantic = batch 
        if cfg.MODEL.CHANNELS == 3:
            input_img = rgb_img
        elif cfg.MODEL.CHANNELS == 4:
            input_img = torch.cat((rgb_img,init_mesh_render_depth),dim=1)
        elif cfg.MODEL.CHANNELS == 5:
            input_img = torch.cat((rgb_img,init_mesh_render_depth,depth_edt),dim=1)
                
        pred_semantic = model(input_img.to(device))["out"].detach().cpu().numpy()[0,:,:]
        pred_semantic = np.argmax(pred_semantic,axis=0)

        for j in range(5):
            count_pixels[j]+=len(torch.where(gt_semantic==j)[0])
        plt.subplot(131)
        plt.imshow(rgb_img.permute(0,2,3,1).numpy()[0,:,:,:])
        plt.subplot(132)
        plt.imshow(gt_semantic.numpy()[0,:,:])
        plt.subplot(133)
        plt.imshow(pred_semantic)
        plt.show()
    total_pixels = torch.sum(count_pixels)
    print(5280*512*512)
    print(count_pixels/total_pixels)