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
from mesh_sem_opt import mesh_sem_opt

cfg_file = "Sensat_single.yaml"
lr = 1e-2
iters = 30

if __name__ == "__main__":

    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)

    worker_id = cfg.SOLVER.GPU_ID
    device = torch.device("cuda:%d" % worker_id)

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

    for seq_idx in ["birmingham_5"]:
        for img_idx in [12]:
            img_idx = "%04d"%img_idx 
            rgb_img, sparse_depth, depth_edt, sem_pred, mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, gt_semantic = load_data_by_index(cfg = cfg, seq_idx = seq_idx,img_idx=img_idx,meshing="mesh1024",samples="1000",device=device)
            mesh = mesh.scale_verts(init_mesh_scale)
            focal_length = cfg.MODEL.MESH_HEAD.FOCAL_LENGTH
            K = [   [focal_length, 0.0, 0.0, 0.0],
                    [0.0, focal_length, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],]
            K = torch.tensor(K)
            P = K[None].repeat(1, 1, 1).to(device).detach()
            vert_pos_padded = project_verts(mesh.verts_padded(), P)
            vert_align_feats = vert_align(sem_pred, vert_pos_padded)
            mesh.textures = TexturesVertex(verts_features=vert_align_feats)
            loss, losses = loss_fn([mesh], None, gt_mesh_pcd, gt_depth, gt_semantic)
            print("chamfer",losses["chamfer_0"])
            print("depth",losses["depth_0"])
            print("semantic",losses["semantic_0"])
            new_mesh = mesh_sem_opt(mesh, sem_pred, lr, iters)
            loss, losses = loss_fn([new_mesh], None, gt_mesh_pcd, gt_depth, gt_semantic)
            print("chamfer",losses["chamfer_0"])
            print("depth",losses["depth_0"])
            print("semantic",losses["semantic_0"])

    '''
    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)

    # Specify the GPU
    worker_id = cfg.SOLVER.GPU_ID
    device = torch.device("cuda:%d" % worker_id)

    # Build the loss
    loss_fn_kwargs = {
        "chamfer_weight": 1e-10,#cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT,
        "depth_weight": 1e-10,#cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT,
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
    #loss_fn.set_semantic_weight(0)

    focal_length = cfg.MODEL.MESH_HEAD.FOCAL_LENGTH
    K = [
                [focal_length, 0.0, 0.0, 0.0],
                [0.0, focal_length, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
    K = torch.tensor(K)
    P = K[None].repeat(1, 1, 1).to(device).detach()

    chamfer_sum = 0
    depth_sum = 0
    sem_sum = 0
    num_img = 0

    #for seq_idx in ["birmingham_3","cambridge_14"]:
    #    for img_idx in range(660):
    for seq_idx in ["birmingham_5"]:
        for img_idx in [12]:
            img_idx = "%04d"%img_idx 

            rgb_img, sparse_depth, depth_edt, sem_pred, mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, gt_semantic = load_data_by_index(cfg = cfg, seq_idx = seq_idx,img_idx=img_idx,meshing="mesh1024",samples="1000",device=device)
            
            vert_pos_padded = project_verts(mesh.verts_padded(), P)
            vert_align_feats = vert_align(sem_pred, vert_pos_padded)
            mesh.textures = TexturesVertex(verts_features=vert_align_feats)
            #mesh.textures = TexturesVertex(verts_features=torch.zeros((1,1024,5), device=device))

            mesh = mesh.scale_verts(init_mesh_scale)
            verts, faces = mesh.get_mesh_verts_faces(0)

            deform_verts = torch.full(
                mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
            sem_verts = torch.full(
                mesh.textures.verts_features_packed().shape, 0.2, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([deform_verts, sem_verts], lr=lr)
            #optimizer = torch.optim.Adam([deform_verts], lr=lr)
            #optimizer = torch.optim.Adam([sem_verts], lr=lr)

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

                loss, losses = loss_fn([new_src_mesh], None, gt_mesh_pcd, gt_depth, gt_semantic)
                loop.set_description('total_loss = %.6f, chamfer_loss = %.6f, depth_loss = %.6f, sem_loss = %.6f.' % (loss, losses["chamfer_0"], losses["depth_0"], losses["semantic_0"]))
                # Optimization step
                #loss.backward()
                #sub_loss = losses["chamfer_0"]+losses["depth_0"]+losses["semantic_0"]
                loss.backward()
                #losses["semantic_0"].backward()
                optimizer.step()    

            #final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
            #save_obj("visualizations/mesh/"+img_idx+".obj", final_verts, final_faces)

            chamfer_sum += losses["chamfer_0"]
            depth_sum += losses["depth_0"]
            sem_sum += losses["semantic_0"]
            num_img += 1

    print("chamfer",chamfer_sum/num_img)
    print("depth",depth_sum/num_img)
    print("semantic",sem_sum/num_img)
    '''



#depth = renderer_depth(new_src_mesh).permute(0,3,1,2).detach().cpu().numpy()
#semantic_predict = renderer_semantic(new_src_mesh).argmax(3)[0,::].detach().cpu().numpy()
#if i%10 == 0:
#    imwrite("visualizations/%d.png"%i,convert_class_to_rgb_sensat_simplified(semantic_predict))
'''
                plt.subplot(231)
                plt.imshow(gt_depth[0,0,::].cpu().numpy())
                plt.subplot(232)
                plt.imshow(depth[0,0,::])
                plt.subplot(233)
                plt.imshow(np.abs(gt_depth[0,0,::].cpu().numpy()-depth[0,0,::]))
                plt.subplot(234)
                plt.imshow(convert_class_to_rgb_sensat_simplified(gt_semantic[0,::].detach().cpu().numpy()))
                plt.subplot(235)
                plt.imshow(convert_class_to_rgb_sensat_simplified(semantic_predict))
                #plt.draw() 
                plt.pause(0.01)
'''