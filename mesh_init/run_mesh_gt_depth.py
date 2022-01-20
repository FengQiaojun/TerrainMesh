# This script is used to generate some meshes initialized by sparse depths measurment. We solve a linear equation.
import os
import numpy as np
from imageio import imread, imwrite
import time
import matplotlib.pyplot as plt
import torch
from pytorch3d.io import save_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)
from pytorch3d.structures import Meshes
from meshing import regular_512_576, regular_512_1024, regular_512_29584, regular_512_90000
from mesh_init_linear_solver import init_mesh, init_mesh_barycentric, MeshRendererWithFragmentsOnly
from mesh_opt import pytorch3d_mesh_dense_opt

dataset_dir = "/mnt/NVMe-2TB/qiaojun/SensatUrban/"
dataset_name_list = ["birmingham_2", "birmingham_3", "birmingham_4", "birmingham_5", "birmingham_6", "cambridge_4",
                     "cambridge_5", "cambridge_6", "cambridge_10", "cambridge_11", "cambridge_12", "cambridge_14", "cambridge_15"]

image_size = 512
cam_c = 256
cam_f = 512
focal_length = -2
depth_scale = 100
num_imgs = 660

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

vertices_dense, faces_dense, laplacian_dense = regular_512_29584()
vertices_dense = (vertices_dense-cam_c)/cam_f
vertices_dense = np.hstack(
    (vertices_dense, np.ones((vertices_dense.shape[0], 1))))

for dataset_name in dataset_name_list:
    print(dataset_name)
    curr_dir = os.path.join(dataset_dir, dataset_name)

    if not os.path.isdir(os.path.join(curr_dir, "Meshes")):
        os.mkdir(os.path.join(curr_dir, "Meshes"))

    for idx in range(num_imgs):
        if os.path.exists(os.path.join(curr_dir, "Meshes", "%04d.obj" % idx)):
            continue
        if (idx%10==0):
            print(idx)
        depth_img_path_read = os.path.join(curr_dir, "Depths", "%04d.png" % idx)
        depth_img = imread(depth_img_path_read)/depth_scale
        depth_img[np.where(depth_img>650)] = 0
        depth_available_map = depth_img>0
        num_depth = np.sum(depth_available_map)
        mean_depth = np.sum(depth_img)/num_depth

        img_loss_depth_full,new_mesh = pytorch3d_mesh_dense_opt(vertices_dense*mean_depth, faces_dense, depth_img, image_size, focal_length, iters=100, return_mesh=True, GPU_id="0", w_laplacian=20, w_edge=0)
        final_verts, final_faces = new_mesh.get_mesh_verts_faces(0)
        # (?) Sometimes invalid result happens
        while not torch.isfinite(final_verts).all():
            img_loss_depth_full,new_mesh = pytorch3d_mesh_dense_opt(vertices_dense*mean_depth, faces_dense, depth_img, image_size, focal_length, iters=100, return_mesh=True, GPU_id="0", w_laplacian=20, w_edge=0)
            final_verts, final_faces = new_mesh.get_mesh_verts_faces(0)
        
        final_obj = os.path.join(curr_dir, "Meshes", "%04d.obj" % idx)
        save_obj(final_obj, final_verts, final_faces)
        points_gt = sample_points_from_meshes(new_mesh, num_samples=10000, return_normals=False)[0].detach().cpu()
        torch.save(points_gt, os.path.join(curr_dir, "Meshes", "%04d_pcd.pt" % idx))
        