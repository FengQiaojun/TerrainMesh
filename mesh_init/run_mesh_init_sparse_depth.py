# This script is used to generate some meshes initialized by sparse depths measurment. We solve a linear equation.
import os
import numpy as np
from imageio import imread, imwrite
import torch
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)
from meshing import regular_512_576, regular_512_1024, regular_512_2025
from mesh_init_linear_solver import init_mesh, init_mesh_barycentric, MeshRendererWithFragmentsOnly

dataset_dir = "/mnt/NVMe-2TB/qiaojun/SensatUrban/"
dataset_name_list = ["birmingham_2", "birmingham_3", "birmingham_4", "birmingham_5", "birmingham_6", "cambridge_4",
                     "cambridge_5", "cambridge_6", "cambridge_10", "cambridge_11", "cambridge_12", "cambridge_14", "cambridge_15"]
sample_num_list = [500, 1000, 2000, 4000]

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

vertices_576, faces_576, laplacian_576 = regular_512_576()
vertices_576 = (vertices_576-cam_c)/cam_f
vertices_576 = np.hstack((vertices_576, np.ones((vertices_576.shape[0], 1))))
pix_to_face_576, bary_coords_576 = init_mesh_barycentric(
    vertices_576, faces_576, image_size, focal_length, device)
vertices_1024, faces_1024, laplacian_1024 = regular_512_1024()
vertices_1024 = (vertices_1024-cam_c)/cam_f
vertices_1024 = np.hstack(
    (vertices_1024, np.ones((vertices_1024.shape[0], 1))))
pix_to_face_1024, bary_coords_1024 = init_mesh_barycentric(
    vertices_1024, faces_1024, image_size, focal_length, device)

vertices_2025, faces_2025, laplacian_2025 = regular_512_2025()
vertices_2025 = (vertices_2025-cam_c)/cam_f
vertices_2025 = np.hstack(
    (vertices_2025, np.ones((vertices_2025.shape[0], 1))))
pix_to_face_2025, bary_coords_2025 = init_mesh_barycentric(
    vertices_2025, faces_2025, image_size, focal_length, device)


# A mesh renderer for depth
cameras = SfMPerspectiveCameras(device=device, focal_length=focal_length,)
raster_settings = RasterizationSettings(
    image_size=image_size,
    blur_radius=0.0001,
    faces_per_pixel=1,
)
rasterizer = MeshRasterizer(
    cameras=cameras,
    raster_settings=raster_settings
)
renderer = MeshRendererWithFragmentsOnly(rasterizer=rasterizer)

#for dataset_name in dataset_name_list:
for dataset_name in ["birmingham_4","cambridge_10","cambridge_11"]:
    print(dataset_name)
    curr_dir = os.path.join(dataset_dir, dataset_name)
    # For each dataset, we need to sample around 500/1000/2000/4000 depth points
    # And for each mesh, we need to work on mesh_576 and mesh_1024

    #for sample_num in sample_num_list:
    for sample_num in [500]:
        if not os.path.isdir(os.path.join(curr_dir, "Pcds_%d" % sample_num)):
            os.mkdir(os.path.join(curr_dir, "Pcds_%d" % sample_num))

    for idx in range(num_imgs):
    #for idx in [139]:
        if (idx%10==0):
            print(idx)
        sparse_depth_img_path_read = os.path.join(
            curr_dir, "Depth_sparse", "%04d.png" % idx)
        sparse_depth_img = imread(sparse_depth_img_path_read)/depth_scale
        sparse_depth_mask = sparse_depth_img > 0
        sparse_depth_idx = np.where(sparse_depth_mask == 1)
        num_sparse_depth = np.sum(sparse_depth_mask)
        index_list = np.arange(num_sparse_depth)
        np.random.shuffle(index_list)
        for sample_num in sample_num_list:
            selected_depth_idx = index_list[:sample_num]
            new_sparse_depth_mask = np.zeros(sparse_depth_mask.shape)
            new_sparse_depth_mask[sparse_depth_idx[0][selected_depth_idx],
                                  sparse_depth_idx[1][selected_depth_idx]] = 1
            new_sparse_depth_img = sparse_depth_img * new_sparse_depth_mask
            '''
            new_vertices_576 = init_mesh(
                new_sparse_depth_img, vertices_576, faces_576, laplacian_576, pix_to_face_576, bary_coords_576)
            expected_verts_576 = torch.tensor(new_vertices_576,dtype=torch.float32,device=device)
            expected_faces_576 = torch.tensor(faces_576,dtype=torch.int32,device=device)
            mesh_576 = Meshes(verts=[expected_verts_576], faces=[expected_faces_576])
            mesh_576.to(device=device)
            fragments = renderer(mesh_576)
            mesh_rendered_depth_576 = fragments.zbuf[0, ..., 0].detach().cpu().numpy()
            
            new_vertices_1024 = init_mesh(
                new_sparse_depth_img, vertices_1024, faces_1024, laplacian_1024, pix_to_face_1024, bary_coords_1024)
            expected_verts_1024 = torch.tensor(new_vertices_1024,dtype=torch.float32,device=device)
            expected_faces_1024 = torch.tensor(faces_1024,dtype=torch.int32,device=device)
            mesh_1024 = Meshes(verts=[expected_verts_1024], faces=[expected_faces_1024])
            mesh_1024.to(device=device)
            fragments = renderer(mesh_1024)
            mesh_rendered_depth_1024 = fragments.zbuf[0, ..., 0].detach().cpu().numpy()
            '''
            new_vertices_2025 = init_mesh(
                new_sparse_depth_img, vertices_2025, faces_2025, laplacian_2025, pix_to_face_2025, bary_coords_2025)
            expected_verts_2025 = torch.tensor(new_vertices_2025,dtype=torch.float32,device=device)
            expected_faces_2025 = torch.tensor(faces_2025,dtype=torch.int32,device=device)
            mesh_2025 = Meshes(verts=[expected_verts_2025], faces=[expected_faces_2025])
            mesh_2025.to(device=device)
            fragments = renderer(mesh_2025)
            mesh_rendered_depth_2025 = fragments.zbuf[0, ..., 0].detach().cpu().numpy()
            
            # We need to store
            # - the sampled points stored as depth image
            # - the initialized mesh
            # - the rendered depth of the initial mesh
            
            #imwrite(os.path.join(curr_dir, "Pcds_%d" % sample_num, "%04d.png"%idx),(new_sparse_depth_img*depth_scale).astype(np.uint16))
            #imwrite(os.path.join(curr_dir, "Pcds_%d" % sample_num, "%04d_mesh576.png"%idx),(mesh_rendered_depth_576*depth_scale).astype(np.uint16))
            #save_obj(os.path.join(curr_dir, "Pcds_%d" % sample_num, "%04d_mesh576.obj"%idx),expected_verts_576,expected_faces_576)
            #imwrite(os.path.join(curr_dir, "Pcds_%d" % sample_num, "%04d_mesh1024.png"%idx),(mesh_rendered_depth_1024*depth_scale).astype(np.uint16))
            #save_obj(os.path.join(curr_dir, "Pcds_%d" % sample_num, "%04d_mesh1024.obj"%idx),expected_verts_1024,expected_faces_1024)
            
            imwrite(os.path.join(curr_dir, "Pcds_%d" % sample_num, "%04d_mesh2025.png"%idx),(mesh_rendered_depth_2025*depth_scale).astype(np.uint16))
            save_obj(os.path.join(curr_dir, "Pcds_%d" % sample_num, "%04d_mesh2025.obj"%idx),expected_verts_2025,expected_faces_2025)
            