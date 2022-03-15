# THe training script
import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from imageio import imwrite

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.io import save_obj

from config import get_sensat_cfg
from dataset.build_data_loader import build_data_loader
from dataset.sensat_dataset import load_data_by_index
from loss import MeshHybridLoss
from mesh_sem_opt import mesh_sem_opt_visualize, mesh_sem_opt_
from model.models import VoxMeshHead
from utils.optimizer import build_optimizer
from utils.model_record_name import generate_model_record_name
from utils.semantic_labels import convert_class_to_rgb_sensat_simplified
from utils.stream_metrics import StreamSegMetrics

cfg_file = "Sensat_predict.yaml"
seq_idx = "cambridge_10"
#img_idx_list = [420]
img_idx_list = range(660)
save_folder = "visualizations/journal/cambridge_10/"

if __name__ == "__main__":
    # Load the config and create a folder to save the outputs.
    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)

    '''
    save_path = generate_model_record_name(cfg, prefix="checkpoints")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    shutil.copyfile(cfg_file, os.path.join(save_path, cfg_file))
    writer = SummaryWriter(os.path.join(save_path))
    '''

    # Specify the GPU
    worker_id = cfg.SOLVER.GPU_ID
    device = torch.device("cuda:%d" % worker_id)


    # Build the model
    if cfg.MODEL.RESUME:
        save_model_path = cfg.MODEL.RESUME_MODEL
        #save_path = cfg.MODEL.RESUME_MODEL.replace("/model_best_depth.tar","")
        save_path = os.path.join(cfg.MODEL.RESUME_MODEL,"..")
        cfg.merge_from_file(os.path.join(save_path,"Sensat_basic.yaml"))
        checkpoint = torch.load(save_model_path)
        # Build the model
        model = VoxMeshHead(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
    else:
        model = VoxMeshHead(cfg)
        model.to(device)     
    model.eval()

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
        "class_weight": cfg.MODEL.DEEPLAB.CLASS_WEIGHT,
        "sem_loss_func": cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_FUNC,
        "device": device
    }
    loss_fn = MeshHybridLoss(**loss_fn_kwargs)

    num_count = 0
    loss_sum = 0
    loss_chamfer_sum = [0]*(cfg.MODEL.MESH_HEAD.NUM_STAGES+1)
    loss_depth_sum = [0]*(cfg.MODEL.MESH_HEAD.NUM_STAGES+1)
    loss_semantic_sum = [0]*(cfg.MODEL.MESH_HEAD.NUM_STAGES+1)

    metrics = StreamSegMetrics(cfg.MODEL.DEEPLAB.NUM_CLASSES)

    for img_idx in img_idx_list:
        print(img_idx)
        img_idx = "%04d"%img_idx 
        rgb_img, sparse_depth, depth_edt, sem_2d_pred, init_mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, gt_semantic = load_data_by_index(cfg = cfg, seq_idx = seq_idx,img_idx=img_idx,meshing="mesh1024",samples="1000",device=device)
            
        # Concatenate the inputs
        if cfg.MODEL.CHANNELS == 3:
            input_img = rgb_img
        elif cfg.MODEL.CHANNELS == 4:
            input_img = torch.cat((rgb_img, init_mesh_render_depth), dim=1)
        elif cfg.MODEL.CHANNELS == 5:
            input_img = torch.cat(
                    (rgb_img, init_mesh_render_depth, depth_edt), dim=1)
        mesh_pred, init_mesh = model(input_img, init_mesh, sem_2d_pred, return_init=True)
        
        #mesh_pred = [init_mesh]
        mesh_pred = [init_mesh]+mesh_pred
        # scale the mesh back to calculate loss
        if cfg.DATASETS.NORMALIZE_MESH:
            init_mesh = init_mesh.scale_verts(init_mesh_scale)
            for m_idx, m in enumerate(mesh_pred):
                mesh_pred[m_idx] = m.scale_verts(init_mesh_scale)

        
        img_gt = convert_class_to_rgb_sensat_simplified(gt_semantic.detach().cpu().numpy()[0,::])
        imwrite(save_folder+seq_idx+"_"+img_idx+"_"+"gt_sem.png",img_gt)
        img_2d = convert_class_to_rgb_sensat_simplified(sem_2d_pred.detach().max(dim=1)[1].cpu().numpy()[0,::])
        imwrite(save_folder+seq_idx+"_"+img_idx+"_"+"2D_sem.png",img_2d)
        
        loss, losses, img_predict = loss_fn(
                init_mesh, None, gt_mesh_pcd, gt_depth, gt_semantic, return_img=True)
        #print("loss_chamfer[0]",losses["chamfer_0"])
        #print("loss_depth[0]",losses["depth_0"])
        #print("loss_semantic[0]",losses["semantic_0"])
        
        img_semantic = img_predict[1].detach().max(dim=1)[1].cpu().numpy()[0,::]
        img_semantic = convert_class_to_rgb_sensat_simplified(img_semantic)
        imwrite(save_folder+seq_idx+"_"+img_idx+"_"+"init_sem.png",img_semantic)
        img_depth = img_predict[0].detach().cpu().numpy()[0,0,::]*100
        imwrite(save_folder+seq_idx+"_"+img_idx+"_"+"init_depth.png",img_depth.astype(np.uint16))
        final_verts, final_faces = init_mesh.get_mesh_verts_faces(0)
        final_obj = seq_idx+"_"+img_idx+"_"+"init.obj"
        save_obj(save_folder+final_obj, final_verts, final_faces)

        metrics.reset()
        metrics.update(img_predict[1].detach().max(dim=1)[1].cpu().numpy(), gt_semantic.cpu().numpy())
        score = metrics.get_results()
        print("Class IoU",score['Class IoU'])          


        loss, losses, img_predict = loss_fn(
                mesh_pred, None, gt_mesh_pcd, gt_depth, gt_semantic, return_img=True)
        #print("loss_chamfer[0]",losses["chamfer_0"])
        #print("loss_depth[0]",losses["depth_0"])
        #print("loss_semantic[0]",losses["semantic_0"])
        
        img_semantic = img_predict[1].detach().max(dim=1)[1].cpu().numpy()[0,::]
        img_semantic = convert_class_to_rgb_sensat_simplified(img_semantic)
        imwrite(save_folder+seq_idx+"_"+img_idx+"_"+"refine_sem.png",img_semantic)
        img_depth = img_predict[0].detach().cpu().numpy()[0,0,::]*100
        imwrite(save_folder+seq_idx+"_"+img_idx+"_"+"refine_depth.png",img_depth.astype(np.uint16))
        final_verts, final_faces = mesh_pred[0].get_mesh_verts_faces(0)
        final_obj = seq_idx+"_"+img_idx+"_"+"refine.obj"
        save_obj(save_folder+final_obj, final_verts, final_faces)

        metrics.reset()
        metrics.update(img_predict[1].detach().max(dim=1)[1].cpu().numpy(), gt_semantic.cpu().numpy())
        score = metrics.get_results()
        print("Class IoU",score['Class IoU'])        

        
        if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
            for i in range(len(mesh_pred)):
                loss_chamfer_sum[i] += losses["chamfer_%d" %
                                                  i].detach().cpu().numpy()*rgb_img.shape[0]
        if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
            for i in range(len(mesh_pred)):
                loss_depth_sum[i] += losses["depth_%d" %
                                                i].detach().cpu().numpy()*rgb_img.shape[0]
        if cfg.MODEL.SEMANTIC and cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT > 0:
            for i in range(len(mesh_pred)):
                loss_semantic_sum[i] += losses["semantic_%d" %
                                                   i].detach().cpu().numpy()*rgb_img.shape[0]

        
        gt_display = gt_depth.cpu().numpy()[0,0,:,:]
        pred_display = init_mesh_render_depth.cpu().numpy()[0,0,:,:]
        depth_available_map = (gt_display>0)*(pred_display>0)
        loss_sum += loss.detach().cpu().numpy()*rgb_img.shape[0]
        num_count += rgb_img.shape[0]


    '''
    if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
        for i in range(len(mesh_pred)):
            print("loss_chamfer_sum[%d]"%i, loss_chamfer_sum[i]/num_count)
    if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
        for i in range(len(mesh_pred)):
            print("loss_depth_sum[%d]"%i, loss_depth_sum[i]/num_count)
    if cfg.MODEL.SEMANTIC and cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT > 0:
        for i in range(len(mesh_pred)):
            print("loss_semantic_sum[%d]"%i, loss_semantic_sum[i]/num_count)
    '''