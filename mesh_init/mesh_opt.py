## A few different ways of meshing
## Maining including the regular-grid like meshing

import numpy as np
import os
from imageio import imread
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from tqdm import tqdm
import torch
import torch.nn as nn
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import (
    mesh_laplacian_smoothing,
    chamfer_distance,
    mesh_edge_loss,
    mesh_normal_consistency
)
from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)
from pytorch3d.io import save_obj, save_ply


class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer):
        super().__init__()
        self.rasterizer = rasterizer

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        return fragments.zbuf

def pytorch3d_mesh_dense_opt(vertices, faces, depth_img, image_size=512, focal_length=-10, iters=100, return_mesh=False, GPU_id="0", w_laplacian=20, w_edge=0, lr=1.0, verbose=False):
    device = torch.device("cuda:"+GPU_id)
    torch.cuda.set_device(device)
    expected_verts = torch.tensor(vertices, dtype=torch.float32, device=device)
    expected_faces = torch.tensor(faces, dtype=torch.int64, device=device)
    mesh = Meshes(verts=[expected_verts], faces=[expected_faces])
    mesh.to(device=device)
    deform_verts = torch.full(
        mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    deform_verts_mask = torch.full(deform_verts.shape, 0.0, device=device)
    deform_verts_mask[:, 2] = 1
    depth_img_gpu = torch.from_numpy(depth_img).to(device=device)
    # Define the camera
    R = torch.eye(3, device=device).reshape((1, 3, 3))
    T = torch.zeros(1, 3, device=device)
    cameras = SfMPerspectiveCameras(
        device=device, R=R, T=T, focal_length=focal_length)
    # Define the Rasterization
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0001,
        faces_per_pixel=1,
    )
    # Define the renderer
    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
    )
    optimizer = torch.optim.Adam([deform_verts], lr=lr)
    loop = tqdm(range(iters), disable=not verbose)
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        # Deform the mesh
        #new_src_mesh = mesh.offset_verts(deform_verts*deform_verts_mask)
        # Alternative way
        vert_direct = mesh.verts_packed() / mesh.verts_packed()[:,2:3]
        vert_deform = vert_direct * deform_verts[:,2:3]
        new_src_mesh = mesh.offset_verts(vert_deform)
        depth = renderer(new_src_mesh)
        depth_available_map = (depth_img_gpu > 0)*(depth[0, ..., 0] > 0)
        loss_depth = torch.sum(torch.abs(
            depth_img_gpu-depth[0, ..., 0])**2*depth_available_map)/torch.sum(depth_available_map)
        # Print the losses
        loss_laplacian = mesh_laplacian_smoothing(
            new_src_mesh, method="uniform")
        loss_edge = mesh_edge_loss(new_src_mesh)
        loop.set_description('depth_loss = %.6f, lap_loss = %.6f, edge_loss = %.6f.' % (
            loss_depth, loss_laplacian, loss_edge))
        # Optimization step
        #print(loss_depth)
        loss = loss_depth + w_laplacian*loss_laplacian + w_edge*loss_edge
        loss.backward()
        optimizer.step()

    depth_img_gpu = torch.from_numpy(depth_img).to(device=device)

    img_loss_depth_full = (torch.abs(
        depth_img_gpu-depth[0, ..., 0])*depth_available_map).detach().cpu().numpy()

    if return_mesh:
        return img_loss_depth_full, new_src_mesh
    else:
        return img_loss_depth_full

def pytorch3d_mesh_dense_chamfer_opt(vertices, faces, depth_img, image_size=512, cam_c=256, cam_f=512, iters=100, return_mesh=False, GPU_id="0", w_laplacian=10, num_samples=20000, lr=1.0, verbose=False):
    device = torch.device("cuda:"+GPU_id)
    torch.cuda.set_device(device)
    expected_verts = torch.tensor(vertices, dtype=torch.float32, device=device)
    expected_faces = torch.tensor(faces, dtype=torch.int64, device=device)
    mesh = Meshes(verts=[expected_verts], faces=[expected_faces])
    mesh.to(device=device)
    deform_verts = torch.full(
        mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    deform_verts_mask = torch.full(deform_verts.shape, 0.0, device=device)
    deform_verts_mask[:, 2] = 1
    optimizer = torch.optim.Adam([deform_verts], lr=lr)
    points_gt = pcd_from_depth(depth_img,cam_c,cam_f,to_pytorch3d=True)
    points_gt.to(device=device)
    loop = tqdm(range(iters), disable=not verbose)
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        # Deform the mesh
        new_src_mesh = mesh.offset_verts(deform_verts*deform_verts_mask)
        points_mesh = sample_points_from_meshes(new_src_mesh, num_samples=num_samples, return_normals=False)
        loss_chamfer, _ = chamfer_distance(points_mesh.cuda(), points_gt.cuda())
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        loop.set_description('chamfer_loss = %.6f, laplacian_loss = %.6f.' % (loss_chamfer, loss_laplacian))
        # Optimization step
        loss = loss_chamfer + w_laplacian*loss_laplacian
        loss.backward()
        optimizer.step()

    loss = loss.detach().cpu().numpy()

    if return_mesh:
        return loss, new_src_mesh
    else:
        return loss

def pcd_from_depth(depth_img, cam_c, cam_f, to_pytorch3d=False):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth_img.shape[1], height=depth_img.shape[0], fx=cam_f, fy=cam_f, cx=cam_c, cy=cam_c)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_img.astype(np.float32)),intrinsic,depth_scale=1)
    if to_pytorch3d:
        pcd = Pointclouds(points = [torch.tensor(np.array(pcd.points), dtype=torch.float32)])
    return pcd

if __name__ == "__main__":
    print("Hello World")
