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
from pytorch3d.structures import (
    Meshes,
)
from pytorch3d.loss import (
    mesh_laplacian_smoothing, 
    chamfer_distance,
    mesh_edge_loss
)
from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    RasterizationSettings,
    DirectionalLights,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader,
    MeshRenderer,
    MeshRasterizer,
)
from pytorch3d.io import save_obj,save_ply
import sys 
#sys.path.append("../utils")
#from utils import read_mesh
#sys.path.append("../meshing")
#from regular_mesh import regular_512_1024,regular_offset_x_512_1024,regular_offset_y_512_1024,features_delaunay

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        
    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        return fragments.zbuf

class MeshRendererWithFragmentsOnly(nn.Module):
    def __init__(self, rasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        
    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        return fragments

def pytorch3d_mesh_pcd_opt(vertices,faces,sparse_depth_img,depth_img,central_map,image_size=512,focal_length=-10,iters=100,return_mesh=False,GPU_id="0"):
    device = torch.device("cuda:"+GPU_id)
    torch.cuda.set_device(device)
    expected_verts = torch.tensor(vertices,dtype=torch.float32,device=device)
    expected_faces = torch.tensor(faces,dtype=torch.int64,device=device)
    mesh = Meshes(verts=[expected_verts], faces=[expected_faces])
    mesh.to(device=device)
    deform_verts = torch.full(mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    deform_verts_mask = torch.full(deform_verts.shape, 0.0, device=device)
    deform_verts_mask[:,2] = 1

    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=512,height=512,fx=2560,fy=2560,cx=256,cy=256)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth=o3d.geometry.Image(sparse_depth_img.astype(np.float32)), intrinsic=cam_intrinsic)
    target_pcd = np.array(pcd.points)
    target_pcd = torch.tensor(target_pcd,dtype=torch.float32,device=device)
    target_pcd = torch.unsqueeze(target_pcd,0)

    optimizer = torch.optim.Adam([deform_verts], lr=0.5)
    loop = tqdm(range(iters))
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        # Deform the mesh
        new_src_mesh = mesh.offset_verts(deform_verts*deform_verts_mask)
        sample_trg = sample_points_from_meshes(new_src_mesh, 5000)
        loss_chamfer, _ = chamfer_distance(sample_trg, target_pcd)
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        loss = loss_chamfer + 0.5*loss_laplacian
        loss.backward()
        loop.set_description('chamfer_loss = %.6f, lap_loss = %.6f.' %(loss_chamfer,loss_laplacian))
        optimizer.step() 
     
    return new_src_mesh
    
def pytorch3d_mesh_from_numpy(vertices,faces,GPU_id="0"):
    device = torch.device("cuda:"+GPU_id)
    torch.cuda.set_device(device)
    expected_verts = torch.tensor(vertices,dtype=torch.float32,device=device)
    expected_faces = torch.tensor(faces,dtype=torch.int64,device=device)
    mesh = Meshes(verts=[expected_verts], faces=[expected_faces])
    mesh.to(device=device)
    return mesh
    
def pytorch3d_mesh_sparse_opt(vertices,faces,sparse_depth_img,depth_img,central_map,image_size=512,focal_length=-10,iters=100,return_mesh=False,GPU_id="0"):
    device = torch.device("cuda:"+GPU_id)
    torch.cuda.set_device(device)
    expected_verts = torch.tensor(vertices,dtype=torch.float32,device=device)
    expected_faces = torch.tensor(faces,dtype=torch.int64,device=device)
    mesh = Meshes(verts=[expected_verts], faces=[expected_faces])
    mesh.to(device=device)
    deform_verts = torch.full(mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    deform_verts_mask = torch.full(deform_verts.shape, 0.0, device=device)
    deform_verts_mask[:,2] = 1
    sparse_depth_img_gpu = torch.from_numpy(sparse_depth_img).to(device=device)
    depth_img_gpu = torch.from_numpy(depth_img).to(device=device)
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
    central_map_gpu = torch.from_numpy(central_map).to(device=device)
    optimizer = torch.optim.Adam([deform_verts], lr=0.5)
    loop = tqdm(range(iters))
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        # Deform the mesh
        new_src_mesh = mesh.offset_verts(deform_verts*deform_verts_mask)
        #rgb, depth = renderer(new_src_mesh)
        depth = renderer(new_src_mesh)
        depth_available_map = (sparse_depth_img_gpu>0)*central_map_gpu
        loss_depth = torch.sum(torch.abs(sparse_depth_img_gpu-depth[0, ..., 0])*depth_available_map*central_map_gpu)/torch.sum(depth_available_map)
        # Print the losses
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        loop.set_description('depth_loss = %.6f, lap_loss = %.6f.' %(loss_depth,loss_laplacian))  
        # Optimization step
        #print(loss_depth)
        loss = loss_depth + 0.5*loss_laplacian
        loss.backward()
        optimizer.step()
     
    depth_img_gpu = torch.from_numpy(depth_img).to(device=device)
    loss_depth_full = torch.sum(torch.abs(depth_img_gpu-depth[0, ..., 0])*central_map_gpu)/torch.sum(central_map_gpu)
    
    img_loss_depth = (torch.abs(sparse_depth_img_gpu-depth[0, ..., 0])*depth_available_map*central_map_gpu).detach().cpu().numpy()
    img_loss_depth_full = (torch.abs(depth_img_gpu-depth[0, ..., 0])*central_map_gpu).detach().cpu().numpy()
    
    #return loss_depth.detach().cpu().numpy(), loss_depth_full.detach().cpu().numpy()
    if return_mesh:
        return img_loss_depth,img_loss_depth_full,new_src_mesh
    else:
        return img_loss_depth,img_loss_depth_full

def pytorch3d_mesh_sparse_opt_mesh_input(mesh,sparse_depth_img,depth_img,central_map,x_offset=0,y_offset=0,image_size=512,focal_length=-10,iters=100,return_mesh=False,GPU_id="0"):
    device = torch.device("cuda:"+GPU_id)
    torch.cuda.set_device(device)
    deform_verts = torch.full(mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    deform_verts_mask = torch.full(deform_verts.shape, 0.0, device=device)
    deform_verts_mask[:,2] = 1
    # decide whether the vertex is in the view
    temp_verts = mesh.verts_packed().clone()
    temp_verts[:,0] -= x_offset
    temp_verts[:,1] -= y_offset
    temp_verts /= torch.unsqueeze(temp_verts[:,2], 1)
    temp_verts = torch.abs(temp_verts)
    visible_verts = (temp_verts[:,0]<0.1) * (temp_verts[:,1]<0.1)
    deform_verts_mask[:,2] *= visible_verts

    sparse_depth_img_gpu = torch.from_numpy(sparse_depth_img).to(device=device)
    depth_img_gpu = torch.from_numpy(depth_img).to(device=device)
    # Define the camera
    R = torch.eye(3,device=device).reshape((1,3,3))
    T = torch.zeros(1,3,device=device)
    T[0,0] = -x_offset
    T[0,1] = -y_offset
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
    central_map_gpu = torch.from_numpy(central_map).to(device=device)
    optimizer = torch.optim.Adam([deform_verts], lr=0.5)
    loop = tqdm(range(iters))
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        # Deform the mesh
        new_src_mesh = mesh.offset_verts(deform_verts*deform_verts_mask)
        #rgb, depth = renderer(new_src_mesh)
        depth = renderer(new_src_mesh)
        depth_available_map = (sparse_depth_img_gpu>0)*central_map_gpu
        loss_depth = torch.sum(torch.abs(sparse_depth_img_gpu-depth[0, ..., 0])*depth_available_map*central_map_gpu)/torch.sum(depth_available_map)
        # Print the losses
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        loop.set_description('depth_loss = %.6f, lap_loss = %.6f.' %(loss_depth,loss_laplacian))  
        # Optimization step
        #print(loss_depth)
        loss = loss_depth + 0.5*loss_laplacian
        loss.backward(retain_graph=True)
        optimizer.step()
     
    depth_img_gpu = torch.from_numpy(depth_img).to(device=device)
    loss_depth_full = torch.sum(torch.abs(depth_img_gpu-depth[0, ..., 0])*central_map_gpu)/torch.sum(central_map_gpu)
    
    img_loss_depth = (torch.abs(sparse_depth_img_gpu-depth[0, ..., 0])*depth_available_map*central_map_gpu).detach().cpu().numpy()
    img_loss_depth_full = (torch.abs(depth_img_gpu-depth[0, ..., 0])*central_map_gpu).detach().cpu().numpy()
    
    #return loss_depth.detach().cpu().numpy(), loss_depth_full.detach().cpu().numpy()
    if return_mesh:
        return img_loss_depth,img_loss_depth_full,new_src_mesh
    else:
        return img_loss_depth,img_loss_depth_full

def pytorch3d_mesh_dense_opt(vertices,faces,depth_img,central_map,image_size=512,focal_length=-10,iters=100,return_mesh=False,GPU_id="0"):
    device = torch.device("cuda:"+GPU_id)
    torch.cuda.set_device(device)
    expected_verts = torch.tensor(vertices,dtype=torch.float32,device=device)
    expected_faces = torch.tensor(faces,dtype=torch.int64,device=device)
    mesh = Meshes(verts=[expected_verts], faces=[expected_faces])
    mesh.to(device=device)
    deform_verts = torch.full(mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    deform_verts_mask = torch.full(deform_verts.shape, 0.0, device=device)
    deform_verts_mask[:,2] = 1
    depth_img_gpu = torch.from_numpy(depth_img).to(device=device)
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
    central_map_gpu = torch.from_numpy(central_map).to(device=device)
    optimizer = torch.optim.Adam([deform_verts], lr=1.0)
    loop = tqdm(range(iters))
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        # Deform the mesh
        new_src_mesh = mesh.offset_verts(deform_verts*deform_verts_mask)
        #rgb, depth = renderer(new_src_mesh)
        depth = renderer(new_src_mesh)
        depth_available_map = (depth_img_gpu>0)*(depth[0, ..., 0]>0)
        loss_depth = torch.sum(torch.abs(depth_img_gpu-depth[0, ..., 0])*depth_available_map)/torch.sum(depth_available_map)
        # Print the losses
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        loss_edge = mesh_edge_loss(new_src_mesh)
        loop.set_description('depth_loss = %.6f, lap_loss = %.6f, edge_loss = %.6f.' %(loss_depth,loss_laplacian,loss_edge))  
        # Optimization step
        #print(loss_depth)
        loss = loss_depth + 0.5*loss_laplacian + 0.1*loss_edge
        loss.backward()
        optimizer.step()

    depth_img_gpu = torch.from_numpy(depth_img).to(device=device)
    loss_depth_full = torch.sum(torch.abs(depth_img_gpu-depth[0, ..., 0])*central_map_gpu)/torch.sum(central_map_gpu)
    
    img_loss_depth_full = (torch.abs(depth_img_gpu-depth[0, ..., 0])*central_map_gpu).detach().cpu().numpy()

    if return_mesh:
        return img_loss_depth_full,new_src_mesh
    else:
        return img_loss_depth_full    


def vis_pytorch3d_mesh_sparse_opt(vertices,faces,sparse_depth_img,depth_img,central_map,image_size=512,focal_length=-10,iters=100,GPU_id="0"):
    device = torch.device("cuda:"+GPU_id)
    torch.cuda.set_device(device)
    expected_verts = [torch.tensor(vert,dtype=torch.float32,device=device) for vert in vertices] 
    expected_faces = [torch.tensor(face,dtype=torch.int64,device=device) for face in faces]
    mesh = Meshes(verts=expected_verts, faces=expected_faces)
    mesh.to(device=device)
    deform_verts = torch.full(mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    deform_verts_mask = torch.full(deform_verts.shape, 0.0, device=device)
    deform_verts_mask[:,2] = 1
    sparse_depth_img_gpu = torch.from_numpy(sparse_depth_img).to(device=device)
    depth_img_gpu = torch.from_numpy(depth_img).to(device=device)
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
    central_map_gpu = torch.from_numpy(central_map).to(device=device)
    optimizer = torch.optim.Adam([deform_verts], lr=0.5)
    loop = tqdm(range(iters))
    for i in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        # Deform the mesh
        new_src_mesh = mesh.offset_verts(deform_verts*deform_verts_mask)
        depth = renderer(new_src_mesh)
        #print(depth.shape)
        depth_available_map = (sparse_depth_img_gpu>0)*central_map_gpu
        loss_depth = torch.sum(torch.abs(sparse_depth_img_gpu-depth[..., 0])*depth_available_map*central_map_gpu)/torch.sum(depth_available_map)
        #depth_available_map = (depth_img_gpu>0)*central_map_gpu
        #loss_depth = torch.sum(torch.abs(depth_img_gpu-depth[..., 0])*depth_available_map*central_map_gpu)/torch.sum(depth_available_map)
        # Print the losses
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        loop.set_description('depth_loss = %.6f, lap_loss = %.6f.' %(loss_depth,loss_laplacian))  
        # Optimization step
        #print(loss_depth)
        loss = loss_depth + 0.5*loss_laplacian
        loss.backward()
        optimizer.step() 

    if True:
        depth_img_gpu = torch.from_numpy(depth_img).to(device=device)
        loss_depth_full = torch.sum(torch.abs(depth_img_gpu-depth[..., 0])*central_map_gpu)/torch.sum(central_map_gpu)
        plt.subplot(131)
        plt.imshow(depth_img)
        plt.subplot(132)
        depth_render_display = (depth[0, ..., 0]*central_map_gpu).detach().cpu().numpy()
        depth_render_display[np.where(depth_render_display<=0)] = np.mean(depth_img)
        plt.imshow(depth_render_display)
        plt.subplot(133)
        plt.imshow((torch.abs(depth_img_gpu-depth[0, ..., 0])*central_map_gpu).detach().cpu().numpy())
        #plt.pause(0.01)
        plt.show()
        

    # Fetch the verts and faces of the final predicted mesh
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
    # Store the predicted mesh using save_obj
    final_obj = os.path.join('final_model.obj')
    save_obj(final_obj, final_verts, final_faces)
    #return loss_depth.detach().cpu().numpy(), loss_depth_full.detach().cpu().numpy()

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

if __name__ == "__main__":
    print("Hello World")

'''
img_idx = "0021"

DATA_DIR = "../data"
rgb_ref = imread(os.path.join(DATA_DIR, img_idx+".png"))

#vertices,faces = regular_512_1024()
#vertices,faces = regular_offset_x_512_1024()
#vertices,faces = regular_offset_y_512_1024()
#vertices,faces = features_delaunay(rgb_ref)
vertices,faces = read_mesh(os.path.join(DATA_DIR,img_idx+"_mesh.txt"))
#print(vertices.shape,faces.shape)

# Convert the vertices to the world frame with a fixed depth.

cam_c = 256
cam_f = 2560
depth_ref = imread(os.path.join(DATA_DIR, img_idx+"_depth.png"))/64
uniform_depth = np.mean(depth_ref[np.where(depth_ref>0)])
#print("uniform_depth",uniform_depth)
vertices = (vertices-cam_c)/cam_f
vertices = np.hstack((vertices,np.ones((vertices.shape[0],1))))
vertices *= uniform_depth

#vertices,faces = read_mesh(os.path.join(DATA_DIR,img_idx+"_sfm_mesh.txt"))
'''


'''
# Fetch the verts and faces of the final predicted mesh
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
# Store the predicted mesh using save_obj
final_obj = os.path.join(DATA_DIR, 'final_model.obj')
save_obj(final_obj, final_verts, final_faces)
'''