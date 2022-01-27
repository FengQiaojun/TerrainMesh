# THe training script
import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    model = VoxMeshHead(cfg)
    model.to(device)
    # Build the optimizer
    optimizer = build_optimizer(cfg, model)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=2, threshold=1e-3)

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
        "focal_length": cfg.MODEL.MESH_HEAD.FOCAL_LENGTH,
        "semantic": cfg.MODEL.SEMANTIC,
        "device": device
    }
    loss_fn = MeshHybridLoss(**loss_fn_kwargs)

    # Build the DataLoaders
    loaders = {}
    loaders["test"] = build_data_loader(
        cfg, "Sensat", split_name=cfg.DATASETS.TESTSET, num_workers=cfg.DATASETS.NUM_THREADS)
    batch_num_test = int(
        np.ceil(len(loaders["test"].dataset)/loaders["test"].batch_size))
    print("Test set size %d. Test batch number %d." %
          (len(loaders["test"].dataset), batch_num_test))

    model.eval()

    num_count = 0
    loss_sum = 0
    loss_chamfer_sum = [0]*cfg.MODEL.MESH_HEAD.NUM_STAGES
    loss_depth_sum = [0]*cfg.MODEL.MESH_HEAD.NUM_STAGES
    loss_semantic_sum = [0]*cfg.MODEL.MESH_HEAD.NUM_STAGES

    loop = tqdm(enumerate(loaders["test"]), total=batch_num_test)
    for i, batch in loop:
        batch = loaders["test"].postprocess(batch, device)
        rgb_img, sparse_depth, depth_edt, init_mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, gt_semantic = batch
        # Concatenate the inputs
        if cfg.MODEL.CHANNELS == 3:
            input_img = rgb_img
        elif cfg.MODEL.CHANNELS == 4:
            input_img = torch.cat((rgb_img, init_mesh_render_depth), dim=1)
        elif cfg.MODEL.CHANNELS == 5:
            input_img = torch.cat(
                    (rgb_img, init_mesh_render_depth, depth_edt), dim=1)
        #mesh_pred = model(input_img, init_mesh)
        mesh_pred = [init_mesh]
        # scale the mesh back to calculate loss
        if cfg.DATASETS.NORMALIZE_MESH:
            for m_idx, m in enumerate(mesh_pred):
                mesh_pred[m_idx] = m.scale_verts(init_mesh_scale)

        loss, losses = loss_fn(
                mesh_pred, gt_mesh_pcd, gt_depth, gt_semantic)
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
        pred_display = init_mesh_render_depth.cpu().numpy()[0,0,:,:]*1000
        depth_available_map = (gt_display>0)*(pred_display>0)
        loss_sum += loss.detach().cpu().numpy()*rgb_img.shape[0]
        num_count += rgb_img.shape[0]

    if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
        for i in range(len(mesh_pred)):
            print("loss_chamfer_sum[%d]"%i, loss_chamfer_sum[i]/num_count)
    if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
        for i in range(len(mesh_pred)):
            print("loss_depth_sum[%d]"%i, loss_depth_sum[i]/num_count)
    if cfg.MODEL.SEMANTIC and cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT > 0:
        for i in range(len(mesh_pred)):
            print("loss_semantic_sum[%d]"%i, loss_semantic_sum[i]/num_count)
    