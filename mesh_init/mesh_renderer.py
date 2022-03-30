# render the mesh using a customized naive shader that preserve the vertex color without any other effects.
import os 
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
import time
import torch
import torch.nn as nn
from pytorch3d.loss import (
    point_mesh_face_distance,
    point_mesh_edge_distance,
    chamfer_distance,
)
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    PerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    TexturesVertex,
)

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments


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

def render_mesh_texture(mesh,image_size=512,focal_length=-1,device=None):
    R = torch.eye(3,device=device).reshape((1,3,3))
    T = torch.zeros(1,3,device=device)
    cameras = SfMPerspectiveCameras(device=device, R=R, T=T, focal_length=focal_length,)
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0001,
        faces_per_pixel=1, 
        perspective_correct=False, # this seems solve the nan error
    )
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    shader = TextureShader(device=device)
    
    renderer = MeshRendererWithFragments(
        rasterizer=rasterizer, shader=shader
    )
    images, depth = renderer(mesh)
    return images.permute(0,3,1,2), depth

def render_mesh_vertex_texture(verts,faces,feats,image_size=512,focal_length=-1,device=None):
    textures = TexturesVertex(verts_features=feats.to(device))
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)])
    mesh.textures = textures
    #mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=textures)
    R = torch.eye(3,device=device).reshape((1,3,3))
    T = torch.zeros(1,3,device=device)
    cameras = SfMPerspectiveCameras(device=device, R=R, T=T, focal_length=focal_length,)
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.00001,
        faces_per_pixel=1, 
    )
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    shader = TextureShader(device=device)
    
    renderer = MeshRendererWithFragments(
        rasterizer=rasterizer, shader=shader
    )
    images, depth = renderer(mesh)
    return images, depth

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        
    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        return fragments.zbuf

def mesh_render_depth(mesh,image_size=512,focal_length=-10,GPU_id="0"):
    device = torch.device("cuda:"+GPU_id)
    # Define the camera
    R = torch.eye(3,device=device).reshape((1,3,3))
    T = torch.zeros(1,3,device=device)
    cameras = SfMPerspectiveCameras(device=device, R=R, T=T, focal_length=focal_length)
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
    depth = renderer(mesh)
    depth = depth[0, ..., 0].detach().cpu().numpy()
    # fill the empty region with mean
    depth_mean = np.mean(depth[np.where(depth>0)])
    depth[np.where(depth<=0)] = depth_mean
    return depth

def pcd_from_depth(depth_img, cam_c, cam_f, to_pytorch3d=False):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth_img.shape[1], height=depth_img.shape[0], fx=cam_f, fy=cam_f, cx=cam_c, cy=cam_c)
    depth_image = depth_img.astype(np.float32)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_img),intrinsic,depth_scale=1)
    if to_pytorch3d:
        pcd = Pointclouds(points = [torch.tensor(np.array(pcd.points), dtype=torch.float32)])
    return pcd

dataset_dir = "/mnt/NVMe-2TB/qiaojun/SensatUrban/"
dataset_name = "cambridge_4"
data_idx = "0000"
cam_c = 256
cam_f = 512

if __name__ == "__main__":
    mesh = o3d.io.read_triangle_mesh(os.path.join(dataset_dir,dataset_name,"Meshes",data_idx+".obj"))
    depth_img = imread(os.path.join(dataset_dir,dataset_name,"Depths",data_idx+".png"))/100
    depth_img = depth_img.astype(np.float32)
    pcd = pcd_from_depth(depth_img, cam_c, cam_f)
    o3d.io.write_point_cloud("0000.ply",pcd)
    o3d.visualization.draw_geometries([pcd,mesh],mesh_show_back_face=True)

    '''
    pcd_pytorch = pcd_from_depth(depth_img,cam_c,cam_f,to_pytorch3d=True)
    pcd_pytorch.to(device=torch.device("cuda:1"))
    mesh_pytorch = Meshes(verts=[torch.tensor(mesh.vertices, dtype=torch.float32)], faces=[torch.tensor(mesh.triangles, dtype=torch.int64)])
    mesh_pytorch.to(device=torch.device("cuda:1"))
    mesh_pcd_pytorch = sample_points_from_meshes(mesh_pytorch, num_samples=10000, return_normals=False)
    cham_loss, _ = chamfer_distance(mesh_pcd_pytorch, pcd_pytorch)
    print(cham_loss)
    '''