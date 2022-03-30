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

from pytorch3d.ops import vert_align
from pytorch3d.renderer import (
    RasterizationSettings,
    SfMPerspectiveCameras,
    SoftPhongShader,
    TexturesVertex,
    MeshRasterizer,
    MeshRenderer,
)

from config import get_sensat_cfg
from dataset.build_data_loader import build_data_loader
from dataset.sensat_dataset import load_data_by_index
from loss import MeshHybridLoss
from mesh_init.mesh_renderer import render_mesh_texture
from model.models import VoxMeshHead
from utils.model_record_name import generate_model_record_name
from utils.optimizer import build_optimizer
from utils.project_verts import project_verts
from utils.semantic_labels import convert_class_to_rgb_sensat_simplified
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
        cfg.merge_from_file(cfg_file)
    else:
        model = VoxMeshHead(cfg)
        model.to(device)     
    model.eval()

    num_count = 0
    loss_sum = 0
    loss_chamfer_sum = [0]*cfg.MODEL.MESH_HEAD.NUM_STAGES
    loss_depth_sum = [0]*cfg.MODEL.MESH_HEAD.NUM_STAGES
    loss_semantic_sum = [0]*cfg.MODEL.MESH_HEAD.NUM_STAGES

    metrics = StreamSegMetrics(cfg.MODEL.DEEPLAB.NUM_CLASSES)
    metrics_single = StreamSegMetrics(cfg.MODEL.DEEPLAB.NUM_CLASSES)

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
        mesh_pred, init_mesh = model(input_img, init_mesh, sem_2d_pred, return_init=True)
        
        if cfg.DATASETS.NORMALIZE_MESH:
            init_mesh = init_mesh.scale_verts(init_mesh_scale)
            for m_idx, m in enumerate(mesh_pred):
                mesh_pred[m_idx] = m.scale_verts(init_mesh_scale)
        '''
        focal_length = cfg.MODEL.MESH_HEAD.FOCAL_LENGTH
        K = [   [focal_length, 0.0, 0.0, 0.0],
                    [0.0, focal_length, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],]
        K = torch.tensor(K)
        P = K[None].repeat(sem_2d_pred.shape[0], 1, 1).to(device).detach()
        vert_pos_padded = project_verts(mesh_pred[1].verts_padded(), P)
        vert_align_feats = vert_align(sem_2d_pred, vert_pos_padded)
        mesh_pred[1].textures = TexturesVertex(verts_features=vert_align_feats)
        '''
        sem_image, depth = render_mesh_texture(mesh_pred[-1],image_size=512,focal_length=-2,device=device)
        metrics.update(sem_image.detach().max(dim=1)[1].cpu().numpy(), gt_semantic.cpu().numpy())
        
        metrics_single.reset()
        metrics_single.update(sem_image.detach().max(dim=1)[1].cpu().numpy(), gt_semantic.cpu().numpy())
        score = metrics_single.get_results()
        print("Class IoU",score['Class IoU'])
        plt.subplot(121)
        plt.imshow(convert_class_to_rgb_sensat_simplified(sem_image.detach().max(dim=1)[1].cpu().numpy()[0,::]))
        plt.subplot(122)
        plt.imshow(convert_class_to_rgb_sensat_simplified(gt_semantic.cpu().numpy()[0,::]))
        plt.show()

    score = metrics.get_results()
    print("GT label distribution", np.sum(metrics.confusion_matrix,axis=1)/np.sum(metrics.confusion_matrix))
    print("Predict label distribution", np.sum(metrics.confusion_matrix,axis=0)/np.sum(metrics.confusion_matrix))
    print("Acc",score['Overall Acc'])
    print("Mean Acc",score['Mean Acc'])
    print("MeanIoU",score['Mean IoU'])
    print("Class IoU",score['Class IoU'])
    print("Recall", np.diag(metrics.confusion_matrix) / metrics.confusion_matrix.sum(axis=1))
    print("Precision", np.diag(metrics.confusion_matrix) / metrics.confusion_matrix.sum(axis=0))