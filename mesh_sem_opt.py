# Test how far we can go if we optimize a single mesh.
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from imageio import imwrite

import torch
import torch.nn as nn
from pytorch3d.io import save_obj
from pytorch3d.ops import vert_align
from pytorch3d.renderer import (
    RasterizationSettings,
    SfMPerspectiveCameras,
    SoftPhongShader,
    TexturesVertex,
    MeshRasterizer,
    MeshRenderer,
)
from pytorch3d.structures import Meshes

from config import get_sensat_cfg
from dataset.sensat_dataset import load_data_by_index
from loss import MeshHybridLoss
from utils.project_verts import project_verts
from utils.semantic_labels import convert_class_to_rgb_sensat_simplified

cfg_file = "Sensat_single.yaml"

# The function to perform mesh semantic optimization
def mesh_sem_opt_visualize(mesh, sem_pred, lr, iters, cfg_file="Sensat_single.yaml"):
    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)

    # Specify the GPU
    worker_id = cfg.SOLVER.GPU_ID
    device = torch.device("cuda:%d" % worker_id)

    # Build the loss
    loss_fn_kwargs = {
        "chamfer_weight": 0,
        "depth_weight": 0,
        "normal_weight": 0,
        "edge_weight": cfg.MODEL.MESH_HEAD.EDGE_LOSS_WEIGHT,
        "laplacian_weight": cfg.MODEL.MESH_HEAD.LAPLACIAN_LOSS_WEIGHT,
        "semantic_weight": cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT,
        "gt_num_samples": cfg.MODEL.MESH_HEAD.GT_NUM_SAMPLES,
        "pred_num_samples": cfg.MODEL.MESH_HEAD.PRED_NUM_SAMPLES,
        "image_size": cfg.MODEL.MESH_HEAD.IMAGE_SIZE,
        "focal_length": cfg.MODEL.MESH_HEAD.FOCAL_LENGTH,
        "class_weight": cfg.MODEL.DEEPLAB.CLASS_WEIGHT,
        "sem_loss_func": cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_FUNC,
        "device": device
    }
    loss_fn = MeshHybridLoss(**loss_fn_kwargs)

    focal_length = cfg.MODEL.MESH_HEAD.FOCAL_LENGTH
    K = [   [focal_length, 0.0, 0.0, 0.0],
            [0.0, focal_length, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],]
    K = torch.tensor(K)
    P = K[None].repeat(1, 1, 1).to(device).detach()

    if mesh.textures is None:
        vert_pos_padded = project_verts(mesh.verts_padded(), P)
        vert_align_feats = vert_align(sem_pred, vert_pos_padded)
        mesh.textures = TexturesVertex(verts_features=vert_align_feats)
    else:
        vert_align_feats = mesh.textures.verts_features_packed()
    #mesh = mesh.scale_verts(init_mesh_scale)
    verts, faces = mesh.get_mesh_verts_faces(0)

    deform_verts = torch.full(
                mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    sem_verts = torch.full(
                mesh.textures.verts_features_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([deform_verts, sem_verts], lr=lr)


    class TextureShader(nn.Module):
        """
        A super simple shader that directly attach texture to the image 
        """
        def __init__(
            self, device="cpu"):
            super().__init__()
            
        def to(self, device):
            # Manually move to device modules which are not subclasses of nn.Module
            return self

        def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
            texels = meshes.sample_textures(fragments)
            colors = texels
            # remove the dimension of K which is the number of triangles/surfaces
            N, H, W, K, C = colors.shape
            if K > 1:
                colors = colors[:,:,:,0,:]
            images = torch.squeeze(colors,dim=3)
            return images

    R = torch.eye(3).reshape((1,3,3))
    R = R.to(device)
    T = torch.zeros(1,3)
    T = T.to(device)
    cameras = SfMPerspectiveCameras(device=device, R=R, T=T, focal_length=-2)
    # Define the Rasterization
    raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=0.0001, 
            faces_per_pixel=1, 
            perspective_correct=False, # this seems solve the nan error
    )
    renderer_semantic = MeshRenderer(
            rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
            ),
        shader = TextureShader(device=device)
    )

    loop = tqdm(range(iters), disable=False)
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        # Deform the mesh
        new_src_mesh = Meshes(
            verts=[verts.to(device)+deform_verts],   
            faces=[faces.to(device)], 
            textures=TexturesVertex(verts_features=vert_align_feats+sem_verts.view(1,1024,5)) 
        )

        #sem_pred_label = sem_pred.argmax(1)
        sem_pred_label = sem_pred
        loss, losses = loss_fn([new_src_mesh], None, None, None, sem_pred_label)
        loop.set_description('total_loss = %.6f, sem_loss = %.6f, lap_loss = %.6f, lap_sem_loss = %.6f.' % (loss, losses["semantic_0"], losses["laplacian_0"], losses["laplacian_sem_0"]))
        loss.backward(retain_graph=True)
        optimizer.step() 

        plt.subplot(121)
        plt.imshow(convert_class_to_rgb_sensat_simplified(sem_pred_label[0,::].detach().cpu().numpy()))
        plt.subplot(122)
        semantic_predict = renderer_semantic(new_src_mesh).permute(0,3,1,2)
        plt.imshow(convert_class_to_rgb_sensat_simplified(semantic_predict.argmax(1)[0,::].detach().cpu().numpy()))
        plt.pause(1)

    #final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
    #save_obj("visualizations/mesh/"+img_idx+".obj", final_verts, final_faces)
    return new_src_mesh

# The function to perform mesh semantic optimization
def mesh_sem_opt_(mesh, sem_pred, gt_mesh_pcd, gt_depth, gt_semantic, lr, iters, cfg_file="Sensat_single.yaml"):
    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)

    # Specify the GPU
    worker_id = cfg.SOLVER.GPU_ID
    device = torch.device("cuda:%d" % worker_id)

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
        "class_weight": cfg.MODEL.DEEPLAB.CLASS_WEIGHT,
        "sem_loss_func": cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_FUNC,
        "device": device
    }
    loss_fn = MeshHybridLoss(**loss_fn_kwargs)

    focal_length = cfg.MODEL.MESH_HEAD.FOCAL_LENGTH
    K = [   [focal_length, 0.0, 0.0, 0.0],
            [0.0, focal_length, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],]
    K = torch.tensor(K)
    P = K[None].repeat(1, 1, 1).to(device).detach()

    if mesh.textures is None:
        vert_pos_padded = project_verts(mesh.verts_padded(), P)
        vert_align_feats = vert_align(sem_pred, vert_pos_padded)
        mesh.textures = TexturesVertex(verts_features=vert_align_feats)
    else:
        vert_align_feats = mesh.textures.verts_features_packed()
    #mesh = mesh.scale_verts(init_mesh_scale)
    verts, faces = mesh.get_mesh_verts_faces(0)

    deform_verts = torch.full(
                mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    sem_verts = torch.full(
                mesh.textures.verts_features_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([deform_verts, sem_verts], lr=lr)

    loop = tqdm(range(iters), disable=False)
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        # Deform the mesh
        new_src_mesh = Meshes(
            verts=[verts.to(device)+deform_verts],   
            faces=[faces.to(device)], 
            textures=TexturesVertex(verts_features=vert_align_feats+sem_verts.view(1,1024,5)) 
        )

        sem_pred_label = sem_pred.argmax(1)
        #sem_pred_label = sem_pred
        loss, losses = loss_fn([new_src_mesh], None, gt_mesh_pcd, gt_depth, gt_semantic)
        loop.set_description('total_loss = %.6f, sem_loss = %.6f.' % (loss, losses["semantic_0"]))
        loss.backward(retain_graph=True)
        optimizer.step()    

    #final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
    #save_obj("visualizations/mesh/"+img_idx+".obj", final_verts, final_faces)
    return new_src_mesh