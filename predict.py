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
from mesh_init.mesh_renderer import render_mesh_texture
from model.models import VoxMeshHead
from utils.optimizer import build_optimizer
from utils.model_record_name import generate_model_record_name
from utils.stream_metrics import StreamSegMetrics

cfg_file = "Sensat_predict.yaml"


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


    
    # Build the DataLoaders
    loaders = {}
    loaders["train"] = build_data_loader(cfg, "Sensat", split_name=cfg.DATASETS.TRAINSET, num_workers=cfg.DATASETS.NUM_THREADS)
    loaders["val"] = build_data_loader(cfg, "Sensat", split_name=cfg.DATASETS.VALSET, num_workers=cfg.DATASETS.NUM_THREADS)
    batch_num_train = int(np.ceil(len(loaders["train"].dataset)/loaders["train"].batch_size))
    batch_num_val = int(np.ceil(len(loaders["val"].dataset)/loaders["val"].batch_size))
    loaders["test"] = build_data_loader(
        cfg, "Sensat", split_name=cfg.DATASETS.TESTSET, num_workers=cfg.DATASETS.NUM_THREADS)
    batch_num_test = int(
        np.ceil(len(loaders["test"].dataset)/loaders["test"].batch_size))
    print("Test set size %d. Test batch number %d." %
          (len(loaders["test"].dataset), batch_num_test))

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

    loop = tqdm(enumerate(loaders["test"]), total=batch_num_test)
    #loop = tqdm(enumerate(loaders["val"]), total=batch_num_val)
    for _, batch in loop:
        batch = loaders["test"].postprocess(batch, device)
        #batch = loaders["val"].postprocess(batch, device)
        rgb_img, sparse_depth, depth_edt, sem_2d_pred, init_mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, gt_semantic = batch
        # Concatenate the inputs
        if cfg.MODEL.CHANNELS == 3:
            input_img = rgb_img
        elif cfg.MODEL.CHANNELS == 4:
            input_img = torch.cat((rgb_img, init_mesh_render_depth), dim=1)
        elif cfg.MODEL.CHANNELS == 5:
            input_img = torch.cat(
                    (rgb_img, init_mesh_render_depth, depth_edt), dim=1)
        elif cfg.MODEL.CHANNELS == 2:
            input_img = torch.cat((init_mesh_render_depth,depth_edt),dim=1)
        mesh_pred = model(input_img, init_mesh, sem_2d_pred)
        mesh_pred = [init_mesh]+mesh_pred
        #mesh_pred = [init_mesh]
        # scale the mesh back to calculate loss
        if cfg.DATASETS.NORMALIZE_MESH:
            init_mesh = init_mesh.scale_verts(init_mesh_scale)
            for m_idx, m in enumerate(mesh_pred):
                mesh_pred[m_idx] = m.scale_verts(init_mesh_scale)

        loss, losses, pred_imgs = loss_fn(
                mesh_pred, None, gt_mesh_pcd, gt_depth, gt_semantic, return_img=True)
        sem_predict = pred_imgs[1]
        metrics.update(sem_predict.detach().max(dim=1)[1].cpu().numpy(), gt_semantic.cpu().numpy())
        
        if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
            for i in range(len(mesh_pred)):
                loss_chamfer_sum[i] += losses["chamfer_%d" %
                                                  i].detach().cpu().numpy()*rgb_img.shape[0]
                #if (losses["chamfer_%d" %i].detach().cpu().numpy() > 10):
                #    print(losses["chamfer_%d" %i].detach().cpu().numpy())                                  
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

    if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
        for i in range(len(mesh_pred)):
            print("loss_depth_sum[%d]"%i, loss_depth_sum[i]/num_count)
    if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
        for i in range(len(mesh_pred)):
            print("loss_chamfer_sum[%d]"%i, loss_chamfer_sum[i]/num_count)
    if cfg.MODEL.SEMANTIC and cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT > 0:
        for i in range(len(mesh_pred)):
            print("loss_semantic_sum[%d]"%i, loss_semantic_sum[i]/num_count)
    
    score = metrics.get_results()
    print("GT label distribution", np.sum(metrics.confusion_matrix,axis=1)/np.sum(metrics.confusion_matrix))
    print("GT label distribution", np.sum(metrics.confusion_matrix,axis=1))
    print("Predict label distribution", np.sum(metrics.confusion_matrix,axis=0)/np.sum(metrics.confusion_matrix))
    print("Predict label distribution", np.sum(metrics.confusion_matrix,axis=0))
    print("Acc",score['Overall Acc'])
    print("Mean Acc",score['Mean Acc'])
    print("MeanIoU",score['Mean IoU'])
    print("Class IoU",score['Class IoU'])
    print("Recall", np.diag(metrics.confusion_matrix) / metrics.confusion_matrix.sum(axis=1))
    print("Precision", np.diag(metrics.confusion_matrix) / metrics.confusion_matrix.sum(axis=0))