# THe training script
import os
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
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
    
    # Specify the GPU
    worker_id = cfg.SOLVER.GPU_ID
    device = torch.device("cuda:%d" % worker_id)

    if cfg.MODEL.RESUME:
        save_path = generate_model_record_name(cfg,prefix="checkpoints")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        shutil.copyfile(cfg_file, os.path.join(save_path,cfg_file))
        writer = SummaryWriter(save_path)
        checkpoint = torch.load(cfg.MODEL.RESUME_MODEL)
        # Build the model
        model = VoxMeshHead(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        # Build the optimizer
        optimizer = build_optimizer(cfg, model)
        if "optimizer_state_dict" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if cfg.SOLVER.SCHEDULER == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, threshold=1e-3)
        elif cfg.SOLVER.SCHEDULER == "StepLR":
            scheduler = StepLR(optimizer, step_size=cfg.SOLVER.SCHEDULER_STEP_SIZE, gamma=cfg.SOLVER.SCHEDULER_GAMMA)
        else:
            scheduler = StepLR(optimizer, step_size=cfg.SOLVER.SCHEDULER_STEP_SIZE, gamma=cfg.SOLVER.SCHEDULER_GAMMA)
        if "scheduler_state_dict" in checkpoint.keys():
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        save_path = generate_model_record_name(cfg,prefix="checkpoints")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        shutil.copyfile(cfg_file, os.path.join(save_path,cfg_file))
        writer = SummaryWriter(save_path)
        # Build the model
        model = VoxMeshHead(cfg)
        model.to(device)
        # Build the optimizer
        optimizer = build_optimizer(cfg, model)
        if cfg.SOLVER.SCHEDULER == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, threshold=1e-3)
        elif cfg.SOLVER.SCHEDULER == "StepLR":
            scheduler = StepLR(optimizer, step_size=cfg.SOLVER.SCHEDULER_STEP_SIZE, gamma=cfg.SOLVER.SCHEDULER_GAMMA)
        else:
            scheduler = StepLR(optimizer, step_size=cfg.SOLVER.SCHEDULER_STEP_SIZE, gamma=cfg.SOLVER.SCHEDULER_GAMMA)

    
    # Build the loss
    loss_fn_kwargs = {
        "chamfer_weight": cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT,
        "depth_weight": cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT,
        "normal_weight": cfg.MODEL.MESH_HEAD.NORMALS_LOSS_WEIGHT,
        "edge_weight": cfg.MODEL.MESH_HEAD.EDGE_LOSS_WEIGHT,
        "laplacian_weight": cfg.MODEL.MESH_HEAD.LAPLACIAN_LOSS_WEIGHT,
        "semantic_weight": cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT,
        "gt_num_samples": cfg.MODEL.MESH_HEAD.GT_NUM_SAMPLES,
        "pred_num_samples": cfg.MODEL.MESH_HEAD.PRED_NUM_SAMPLES,
        "image_size": cfg.MODEL.MESH_HEAD.IMAGE_SIZE,
        "focal_length": cfg.MODEL.MESH_HEAD.FOCAL_LENGTH,
        "semantic": cfg.MODEL.SEMANTIC,
        "graph_conv_semantic": cfg.MODEL.MESH_HEAD.GRAPH_CONV_SEMANTIC,
        "device": device
    }
    loss_fn = MeshHybridLoss(**loss_fn_kwargs)
    loss_fn.set_semantic_weight(0)
    
    # Build the DataLoaders 
    loaders = {}
    loaders["train"] = build_data_loader(cfg, "Sensat", split_name=cfg.DATASETS.TRAINSET, num_workers=cfg.DATASETS.NUM_THREADS)
    loaders["val"] = build_data_loader(cfg, "Sensat", split_name=cfg.DATASETS.VALSET, num_workers=cfg.DATASETS.NUM_THREADS)
    batch_num_train = int(np.ceil(len(loaders["train"].dataset)/loaders["train"].batch_size))
    batch_num_val = int(np.ceil(len(loaders["val"].dataset)/loaders["val"].batch_size))
    print("Training set size %d. Training batch number %d."%(len(loaders["train"].dataset),batch_num_train))
    print("Validation set size %d. Validation batch number %d."%(len(loaders["val"].dataset),batch_num_val))

    min_chamfer_error = 100
    min_chamfer_epoch = -1
    min_depth_error = 100
    min_depth_epoch = -1
    min_semantic_error = 100
    min_semantic_epoch = -1

    # Start the training epochs
    starting_epoch = checkpoint["epoch"] if cfg.MODEL.RESUME else 0
    for epoch in range(starting_epoch, cfg.SOLVER.NUM_EPOCHS+1):
    
        if cfg.MODEL.SEMANTIC and epoch == cfg.SOLVER.SEM_START_EPOCH:
            loss_fn.set_semantic_weight(cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT)
            model.set_semantic(True)
            print("Switch to semantic mode at epoch ",epoch)
        
        num_count = 0
        loss_sum = 0
        loss_chamfer_sum = [0]*cfg.MODEL.MESH_HEAD.NUM_STAGES
        loss_depth_sum = [0]*cfg.MODEL.MESH_HEAD.NUM_STAGES
        loss_semantic_sum = [0]*cfg.MODEL.MESH_HEAD.NUM_STAGES
        model.train()
        loop = tqdm(enumerate(loaders["train"]),total=batch_num_train)
        for idx, batch in loop:
            batch = loaders["train"].postprocess(batch, device)
            rgb_img, sparse_depth, depth_edt, sem_2d_pred, init_mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, gt_semantic = batch 
            # Concatenate the inputs
            if cfg.MODEL.CHANNELS == 3:
                input_img = rgb_img
            elif cfg.MODEL.CHANNELS == 4:
                input_img = torch.cat((rgb_img,init_mesh_render_depth),dim=1)
            elif cfg.MODEL.CHANNELS == 5:
                input_img = torch.cat((rgb_img,init_mesh_render_depth,depth_edt),dim=1)
            elif cfg.MODEL.CHANNELS == 9:
                input_img = torch.cat((rgb_img,init_mesh_render_depth,depth_edt,sem_2d_pred[:,1:5,::]),dim=1)
            mesh_pred = model(input_img, init_mesh, sem_2d_pred)
            # scale the mesh back to calculate loss
            if cfg.DATASETS.NORMALIZE_MESH:
                for m_idx, m in enumerate(mesh_pred):
                    mesh_pred[m_idx] = m.scale_verts(init_mesh_scale)
            
            loss, losses = loss_fn(mesh_pred, None, gt_mesh_pcd, gt_depth, gt_semantic)
                
            if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
                for i in range(cfg.MODEL.MESH_HEAD.NUM_STAGES):
                    writer.add_scalar("Loss/train/batch/chamfer_%d"%i, losses["chamfer_%d"%i], epoch*batch_num_train+i)
                    loss_chamfer_sum[i] += losses["chamfer_%d"%i].detach().cpu().numpy()*rgb_img.shape[0]
            if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
                for i in range(cfg.MODEL.MESH_HEAD.NUM_STAGES):
                    writer.add_scalar("Loss/train/batch/depth_%d"%i, losses["depth_%d"%i], epoch*batch_num_train+i)
                    loss_depth_sum[i] += losses["depth_%d"%i].detach().cpu().numpy()*rgb_img.shape[0]
            if loss_fn.get_semantic_weight() > 0:
                for i in range(cfg.MODEL.MESH_HEAD.NUM_STAGES):
                    writer.add_scalar("Loss/train/batch/semantic_%d"%i, losses["semantic_%d"%i], epoch*batch_num_train+i)
                    loss_semantic_sum[i] += losses["semantic_%d"%i].detach().cpu().numpy()*rgb_img.shape[0]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train/batch", loss, epoch*batch_num_train+i)
            loss_sum += loss.detach().cpu().numpy()*rgb_img.shape[0]
            num_count += rgb_img.shape[0]
            
        writer.add_scalar("Loss/train/epoch", loss_sum/num_count, epoch)   
        if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
            for i in range(cfg.MODEL.MESH_HEAD.NUM_STAGES):
                writer.add_scalar("Loss/train/epoch/chamfer_%d"%i, loss_chamfer_sum[i]/num_count, epoch)   
        if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
            for i in range(cfg.MODEL.MESH_HEAD.NUM_STAGES):
                writer.add_scalar("Loss/train/epoch/depth_%d"%i, loss_depth_sum[i]/num_count, epoch)  
        if loss_fn.get_semantic_weight() > 0:
            for i in range(cfg.MODEL.MESH_HEAD.NUM_STAGES):
                writer.add_scalar("Loss/train/epoch/semantic_%d"%i, loss_semantic_sum[i]/num_count, epoch)  
                
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, save_path+"/model_latest.tar")



        # Validation
        if (epoch)%cfg.SOLVER.LOGGING_PERIOD == 0 or epoch == cfg.SOLVER.NUM_EPOCHS:
            num_count = 0
            loss_sum = 0
            loss_chamfer_sum = [0]*cfg.MODEL.MESH_HEAD.NUM_STAGES
            loss_depth_sum = [0]*cfg.MODEL.MESH_HEAD.NUM_STAGES
            loss_semantic_sum = [0]*cfg.MODEL.MESH_HEAD.NUM_STAGES
            model.eval()
            for idx, batch in tqdm(enumerate(loaders["val"]),total=batch_num_val):
                batch = loaders["val"].postprocess(batch, device)
                rgb_img, sparse_depth, depth_edt, sem_2d_pred, init_mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, gt_semantic = batch 
                # Concatenate the inputs
                if cfg.MODEL.CHANNELS == 3:
                    input_img = rgb_img
                elif cfg.MODEL.CHANNELS == 4:
                    input_img = torch.cat((rgb_img,init_mesh_render_depth),dim=1)
                elif cfg.MODEL.CHANNELS == 5:
                    input_img = torch.cat((rgb_img,init_mesh_render_depth,depth_edt),dim=1)
                elif cfg.MODEL.CHANNELS == 9:
                    input_img = torch.cat((rgb_img,init_mesh_render_depth,depth_edt,sem_2d_pred[:,1:5,::]),dim=1)
                mesh_pred = model(input_img, init_mesh, sem_2d_pred)
                # scale the mesh back to calculate loss
                if cfg.DATASETS.NORMALIZE_MESH:
                    for m_idx, m in enumerate(mesh_pred):
                        mesh_pred[m_idx] = m.scale_verts(init_mesh_scale)

                loss, losses = loss_fn(mesh_pred, None, gt_mesh_pcd, gt_depth, gt_semantic)
                loss_sum += loss.detach().cpu().numpy()*rgb_img.shape[0]
                
                if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
                    for i in range(cfg.MODEL.MESH_HEAD.NUM_STAGES):
                        loss_chamfer_sum[i] += losses["chamfer_%d"%i].detach().cpu().numpy()*rgb_img.shape[0]
                if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
                    for i in range(cfg.MODEL.MESH_HEAD.NUM_STAGES):
                        loss_depth_sum[i] += losses["depth_%d"%i].detach().cpu().numpy()*rgb_img.shape[0]
                if loss_fn.get_semantic_weight() > 0:
                    for i in range(cfg.MODEL.MESH_HEAD.NUM_STAGES):
                        loss_semantic_sum[i] += losses["semantic_%d"%i].detach().cpu().numpy()*rgb_img.shape[0]
                num_count += rgb_img.shape[0]

            if cfg.SOLVER.SCHEDULER == "ReduceLROnPlateau":
                scheduler.step(loss_sum/num_count)
            elif cfg.SOLVER.SCHEDULER == "StepLR":
                scheduler.step()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, save_path+"/model_%d.tar"%(epoch))
            
            writer.add_scalar("Loss/val/epoch", loss_sum/num_count, epoch)
            if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
                for i in range(cfg.MODEL.MESH_HEAD.NUM_STAGES):
                    writer.add_scalar("Loss/val/epoch/chamfer_%d"%i, loss_chamfer_sum[i]/num_count, epoch)   
                if loss_chamfer_sum[cfg.MODEL.MESH_HEAD.NUM_STAGES-1]/num_count < min_chamfer_error:
                    min_chamfer_error = loss_chamfer_sum[cfg.MODEL.MESH_HEAD.NUM_STAGES-1]/num_count
                    min_chamfer_epoch = epoch
                    shutil.copyfile(save_path+"/model_%d.tar"%(epoch), save_path+"/model_best_chamfer.tar")
            if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
                for i in range(cfg.MODEL.MESH_HEAD.NUM_STAGES):
                    writer.add_scalar("Loss/val/epoch/depth_%d"%i, loss_depth_sum[i]/num_count, epoch)   
                if loss_depth_sum[cfg.MODEL.MESH_HEAD.NUM_STAGES-1]/num_count < min_depth_error:
                    min_depth_error = loss_depth_sum[cfg.MODEL.MESH_HEAD.NUM_STAGES-1]/num_count
                    min_depth_epoch = epoch
                    shutil.copyfile(save_path+"/model_%d.tar"%(epoch), save_path+"/model_best_depth.tar")            
            if loss_fn.get_semantic_weight() > 0:
                for i in range(cfg.MODEL.MESH_HEAD.NUM_STAGES):
                    writer.add_scalar("Loss/val/epoch/semantic_%d"%i, loss_semantic_sum[i]/num_count, epoch)                     
                if loss_semantic_sum[cfg.MODEL.MESH_HEAD.NUM_STAGES-1]/num_count < min_semantic_error:
                    min_semantic_error = loss_semantic_sum[cfg.MODEL.MESH_HEAD.NUM_STAGES-1]/num_count
                    min_semantic_epoch = epoch
                    shutil.copyfile(save_path+"/model_%d.tar"%(epoch), save_path+"/model_best_semantic.tar")
            
            print("Best Chamfer Epoch %d, Best Depth Epoch %d, Best Semantic Epoch %d."%(min_chamfer_epoch,min_depth_epoch,min_semantic_epoch))  
