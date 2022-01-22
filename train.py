# THe training script
import os
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from config import get_sensat_cfg
from dataset.build_data_loader import build_data_loader
from loss import MeshHybridLoss
from model.models import VoxMeshHead
from utils.optimizer import build_optimizer
from utils.model_record_name import generate_model_record_name



cfg_file = "Sensat_basic.yaml"



if __name__ == "__main__":
    # Load the config and create a folder to save the outputs.
    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)

    cfg.DATASETS.SHUFFLE = False
    
    save_path = generate_model_record_name(cfg,prefix="checkpoints")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    shutil.copyfile(cfg_file, os.path.join(save_path,cfg_file))
    writer = SummaryWriter(os.path.join(save_path,"logs"))

    # Specify the GPU
    worker_id = cfg.SOLVER.GPU_ID
    device = torch.device("cuda:%d" % worker_id)

    # Build the model
    model = VoxMeshHead(cfg)
    model.to(device)
    # Build the optimizer
    optimizer = build_optimizer(cfg, model)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, threshold=1e-3)

    # Build the loss
    loss_fn_kwargs = {
        "chamfer_weight": cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT,
        "depth_weight": cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT,
        "normal_weight": cfg.MODEL.MESH_HEAD.NORMALS_LOSS_WEIGHT,
        "edge_weight": cfg.MODEL.MESH_HEAD.EDGE_LOSS_WEIGHT,
        "laplacian_weight": cfg.MODEL.MESH_HEAD.LAPLACIAN_LOSS_WEIGHT,
        "semantic_weight": cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT,
        "gt_num_samples": cfg.MODEL.MESH_HEAD.GT_NUM_SAMPLES,
        "pred_num_samples": cfg.MODEL.MESH_HEAD.PRED_NUM_SAMPLES,
        "image_size": cfg.MODEL.MESH_HEAD.IMAGE_SIZE,
        "device": device
    }
    loss_fn = MeshHybridLoss(**loss_fn_kwargs)

    # Build the DataLoaders 
    loaders = {}
    loaders["train"] = build_data_loader(cfg, "Sensat", "train", num_workers=cfg.DATASETS.NUM_THREADS)
    loaders["val"] = build_data_loader(cfg, "Sensat", "val", num_workers=cfg.DATASETS.NUM_THREADS)
    batch_num_train = int(np.ceil(len(loaders["train"].dataset)/loaders["train"].batch_size))
    batch_num_val = int(np.ceil(len(loaders["val"].dataset)/loaders["val"].batch_size))
    print("Training set size %d. Training batch number %d."%(len(loaders["train"].dataset),batch_num_train))
    print("Validation set size %d. Validation batch number %d."%(len(loaders["val"].dataset),batch_num_val))

    min_chamfer_error = 100
    min_depth_error = 100
    min_semantic_error = 100

    # Start the training epochs
    for epoch in range(cfg.SOLVER.NUM_EPOCHS):
    #for epoch in range(10):

        # Validation
        
        if (epoch)%10 == 0 or epoch+1 == cfg.SOLVER.NUM_EPOCHS:
            num_count = 0
            loss_sum = 0
            loss_chamfer_sum = 0
            loss_depth_sum = 0
            loss_semantic_sum = 0
            model.eval()
            for i, batch in tqdm(enumerate(loaders["val"]),total=batch_num_val):
                batch = loaders["val"].postprocess(batch, device)
                
                rgb_img, sparse_depth, depth_edt, init_mesh, init_mesh_render_depth, gt_depth, gt_mesh_pcd, gt_semantic = batch 
                # Concatenate the inputs
                if cfg.MODEL.CHANNELS == 3:
                    input_img = rgb_img
                elif cfg.MODEL.CHANNELS == 4:
                    input_img = torch.cat((rgb_img,init_mesh_render_depth),dim=1)
                elif cfg.MODEL.CHANNELS == 5:
                    input_img = torch.cat((rgb_img,init_mesh_render_depth,depth_edt),dim=1)
                mesh_pred = model(input_img, init_mesh)
        
                loss, losses = loss_fn(mesh_pred, gt_mesh_pcd, gt_depth, gt_semantic)
                loss_sum += loss.detach().cpu().numpy()*rgb_img.shape[0]
                
                if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
                    loss_chamfer_sum += losses["chamfer_%d"%(cfg.MODEL.MESH_HEAD.NUM_STAGES-1)].detach().cpu().numpy()*rgb_img.shape[0]
                if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
                    loss_depth_sum += losses["depth_%d"%(cfg.MODEL.MESH_HEAD.NUM_STAGES-1)].detach().cpu().numpy()*rgb_img.shape[0]
                if cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT > 0:
                    loss_semantic_sum += losses["semantic_%d"%(cfg.MODEL.MESH_HEAD.NUM_STAGES-1)].detach().cpu().numpy()*rgb_img.shape[0]
                num_count += rgb_img.shape[0]

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_path+"/model_%d.tar"%(epoch))
            
            writer.add_scalar("Loss/val/epoch", loss_sum/num_count, epoch)
            if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
                writer.add_scalar("Loss/val/epoch/chamfer", loss_chamfer_sum/num_count, epoch)   
                if loss_chamfer_sum/num_count < min_chamfer_error:
                    min_chamfer_error = loss_chamfer_sum/num_count
                    min_chamfer_epoch = epoch
                    shutil.copyfile(save_path+"/model_%d.tar"%(epoch), save_path+"/model_best_chamfer.tar")
            if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
                writer.add_scalar("Loss/val/epoch/depth", loss_depth_sum/num_count, epoch)   
                if loss_depth_sum/num_count < min_depth_error:
                    min_depth_error = loss_depth_sum/num_count
                    min_depth_epoch = epoch
                    shutil.copyfile(save_path+"/model_%d.tar"%(epoch), save_path+"/model_best_depth.tar")            
            if cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT > 0:
                writer.add_scalar("Loss/val/epoch/semantic", loss_semantic_sum/num_count, epoch)                     
                if loss_semantic_sum/num_count < min_semantic_error:
                    min_semantic_error = loss_semantic_sum/num_count
                    min_semantic_epoch = epoch
                    shutil.copyfile(save_path+"/model_%d.tar"%(epoch), save_path+"/model_best_semantic.tar")
              

        num_count = 0
        loss_sum = 0
        loss_chamfer_sum = 0
        loss_depth_sum = 0
        loss_semantic_sum = 0
        model.train()
        loop = tqdm(enumerate(loaders["train"]),total=batch_num_train)
        for i, batch in loop:
            batch = loaders["train"].postprocess(batch, device)
            rgb_img, sparse_depth, depth_edt, init_mesh, init_mesh_render_depth, gt_depth, gt_mesh_pcd, gt_semantic = batch 
            # Concatenate the inputs
            if cfg.MODEL.CHANNELS == 3:
                input_img = rgb_img
            elif cfg.MODEL.CHANNELS == 4:
                input_img = torch.cat((rgb_img,init_mesh_render_depth),dim=1)
            elif cfg.MODEL.CHANNELS == 5:
                input_img = torch.cat((rgb_img,init_mesh_render_depth,depth_edt),dim=1)
            mesh_pred = model(input_img, init_mesh)
    
            loss, losses = loss_fn(mesh_pred, gt_mesh_pcd, gt_depth, gt_semantic)
            if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
                writer.add_scalar("Loss/train/batch/chamfer", losses["chamfer_%d"%(cfg.MODEL.MESH_HEAD.NUM_STAGES-1)], epoch*batch_num_train+i)
                loss_chamfer_sum += losses["chamfer_%d"%(cfg.MODEL.MESH_HEAD.NUM_STAGES-1)].detach().cpu().numpy()*rgb_img.shape[0]
            if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
                writer.add_scalar("Loss/train/batch/depth", losses["depth_%d"%(cfg.MODEL.MESH_HEAD.NUM_STAGES-1)], epoch*batch_num_train+i)
                loss_depth_sum += losses["depth_%d"%(cfg.MODEL.MESH_HEAD.NUM_STAGES-1)].detach().cpu().numpy()*rgb_img.shape[0]
            if cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT > 0:
                writer.add_scalar("Loss/train/batch/semantic", losses["semantic_%d"%(cfg.MODEL.MESH_HEAD.NUM_STAGES-1)], epoch*batch_num_train+i)
                loss_semantic_sum += losses["semantic_%d"%(cfg.MODEL.MESH_HEAD.NUM_STAGES-1)].detach().cpu().numpy()*rgb_img.shape[0]

            loop.set_description('chamfer_0 = %.6f, depth_0 = %.6f, edge_0 = %.6f, laplacian_0 = %.6f' % (losses["chamfer_0"],losses["depth_0"],losses["edge_0"],losses["laplacian_0"]))
            if cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT > 0:
                loop.set_description('chamfer_0 = %.6f, depth_0 = %.6f, semantic_0 = %.6f, edge_0 = %.6f, laplacian_0 = %.6f' % (losses["chamfer_0"],losses["depth_0"],losses["semantic_0"],losses["edge_0"],losses["laplacian_0"]))
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train/batch", loss, epoch*batch_num_train+i)
            loss_sum += loss.detach().cpu().numpy()*rgb_img.shape[0]
            num_count += rgb_img.shape[0]
            
        writer.add_scalar("Loss/train/epoch", loss_sum/num_count, epoch)   
        if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
            writer.add_scalar("Loss/train/epoch/chamfer", loss_chamfer_sum/num_count, epoch)   
        if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
            writer.add_scalar("Loss/train/epoch/depth", loss_depth_sum/num_count, epoch)   
        if cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT > 0:
            writer.add_scalar("Loss/train/epoch/semantic", loss_semantic_sum/num_count, epoch)  
        