# Load a PyTorch deeplab and train

import numpy as np
import os
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import get_sensat_cfg
from dataset.build_data_loader import build_data_loader
from utils.model_record_name import generate_segmodel_record_name

cfg_file = "Sensat_basic.yaml"

if __name__ == "__main__":
    # Load the config and create a folder to save the outputs.
    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)

    save_path = generate_segmodel_record_name(cfg,prefix="checkpoints")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    shutil.copyfile(cfg_file, os.path.join(save_path,cfg_file))
    writer = SummaryWriter(os.path.join(save_path))

    # Specify the GPU. Here use 2 of them.
    #worker_id = cfg.SOLVER.GPU_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build the model
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 5, kernel_size=1, stride=1)
      
    model.backbone.conv1 = nn.Conv2d(cfg.MODEL.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = nn.DataParallel(model)
    model.to(device)

    # Build the optimizer
    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.module.backbone.parameters(), 'lr': 0.1 * cfg.MODEL.DEEPLAB.LR},
        {'params': model.module.classifier.parameters(), 'lr': cfg.MODEL.DEEPLAB.LR},
    ], lr=cfg.MODEL.DEEPLAB.LR, momentum=cfg.MODEL.DEEPLAB.MOMENTUM, weight_decay=cfg.MODEL.DEEPLAB.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.MODEL.DEEPLAB.SCHEDULER_STEP_SIZE, gamma=cfg.MODEL.DEEPLAB.SCHEDULER_GAMMA)

    # Set up criterion
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

    # Build the DataLoaders 
    loaders = {}
    loaders["train"] = build_data_loader(cfg, "SensatSemantic", split_name=cfg.DATASETS.TRAINSET, num_workers=cfg.DATASETS.NUM_THREADS)
    loaders["val"] = build_data_loader(cfg, "SensatSemantic", split_name=cfg.DATASETS.VALSET, num_workers=cfg.DATASETS.NUM_THREADS)
    batch_num_train = int(np.ceil(len(loaders["train"].dataset)/loaders["train"].batch_size))
    batch_num_val = int(np.ceil(len(loaders["val"].dataset)/loaders["val"].batch_size))
    print("Training set size %d. Training batch number %d."%(len(loaders["train"].dataset),batch_num_train))
    print("Validation set size %d. Validation batch number %d."%(len(loaders["val"].dataset),batch_num_val))
    
    min_error = 100
    min_epoch = -1
    # Start the training epochs
    for epoch in range(cfg.MODEL.DEEPLAB.NUM_EPOCHS):

        # Validation
        if (epoch)%10 == 0 or epoch+1 == cfg.MODEL.DEEPLAB.NUM_EPOCHS:
            model.eval()
            loss_sum = 0
            num_count = 0
            for i, batch in tqdm(enumerate(loaders["val"]),total=batch_num_val):
                
                rgb_img, sparse_depth, depth_edt, init_mesh_render_depth, gt_semantic = batch 
                # Concatenate the inputs
                if cfg.MODEL.CHANNELS == 3:
                    input_img = rgb_img
                elif cfg.MODEL.CHANNELS == 4:
                    input_img = torch.cat((rgb_img,init_mesh_render_depth),dim=1)
                elif cfg.MODEL.CHANNELS == 5:
                    input_img = torch.cat((rgb_img,init_mesh_render_depth,depth_edt),dim=1)
                input_img = input_img.to(device, dtype=torch.float32)
                gt_semantic = gt_semantic.to(device, dtype=torch.long)
                pred_semantic = model(input_img)["out"]                
                loss = loss_fn(pred_semantic, gt_semantic)
                loss_sum += loss.detach().cpu().numpy()*rgb_img.shape[0]
                num_count += rgb_img.shape[0]

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_path+"/model_%d.tar"%(epoch))
            
            writer.add_scalar("Loss/val/epoch", loss_sum/num_count, epoch)
            if loss_sum/num_count < min_error:
                min_error = loss_sum/num_count
                min_epoch = epoch
                shutil.copyfile(save_path+"/model_%d.tar"%(epoch), save_path+"/model_best_chamfer.tar")

            print("Best Epoch %d."%min_epoch)  

        num_count = 0
        loss_sum = 0
        model.train()
        loop = tqdm(enumerate(loaders["train"]),total=batch_num_train)
        for i, batch in loop:
            rgb_img, sparse_depth, depth_edt, init_mesh_render_depth, gt_semantic = batch 
            # Concatenate the inputs
            if cfg.MODEL.CHANNELS == 3:
                input_img = rgb_img
            elif cfg.MODEL.CHANNELS == 4:
                input_img = torch.cat((rgb_img,init_mesh_render_depth),dim=1)
            elif cfg.MODEL.CHANNELS == 5:
                input_img = torch.cat((rgb_img,init_mesh_render_depth,depth_edt),dim=1)
            input_img = input_img.to(device, dtype=torch.float32)
            gt_semantic = gt_semantic.to(device, dtype=torch.long)
            pred_semantic = model(input_img)["out"]
            loss = loss_fn(pred_semantic, gt_semantic)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train/batch", loss, epoch*batch_num_train+i)
            loss_sum += loss.detach().cpu().numpy()*rgb_img.shape[0]
            num_count += rgb_img.shape[0]
            
        writer.add_scalar("Loss/train/epoch", loss_sum/num_count, epoch)   