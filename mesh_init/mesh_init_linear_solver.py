import numpy as np
import open3d as o3d
from imageio import imread
import os
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import cvxpy as cp
import time
import torch
import torch.nn as nn
from pytorch3d.structures import (
    Meshes,
)
from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    MeshRasterizer,
    PointLights,
)
from mesh_opt import MeshRendererWithFragmentsOnly

from meshing import regular_512_576, regular_512_1024

# The function to init a mesh with sparse depth.
# Here we take inputs including
# - sparse_depth_img
# - vertices: normalized mesh with z-coordinate being 1
# - faces
# - laplacian
# - pix_to_face: given the pixel coordinate, return the index of face
# - bary_coords: given the pixel coordinate, return the barycentric coordinate (but need the face idx to check the vertex indices).
def init_mesh(sparse_depth_img, vertices, faces, laplacian, pix_to_face, bary_coords):
    sparse_depth_mask = sparse_depth_img>0
    num_sparse_depth = np.sum(sparse_depth_mask)
    num_vertices = vertices.shape[0]
    A = torch.zeros(num_sparse_depth+num_vertices,num_vertices)
    b = torch.zeros(num_sparse_depth+num_vertices)
    sparse_depth_idx = np.where(sparse_depth_mask==1)
    for i in range(num_sparse_depth):
        d_idx = [sparse_depth_idx[0][i],sparse_depth_idx[1][i]]
        face_idx = pix_to_face[d_idx[0],d_idx[1]]
        v_idx = faces[face_idx]
        bary_c = bary_coords[d_idx[0],d_idx[1],:]
        A[i,v_idx] = bary_c
        b[i] = sparse_depth_img[d_idx[0],d_idx[1]]
    A[-num_vertices:,:] = laplacian*0.5
    p, _ = torch.lstsq(torch.tensor(b).unsqueeze(1),torch.tensor(A))
    new_vertices = vertices * p.numpy()[:num_vertices,:]
    return new_vertices


if __name__ == "__main__":

    image_size = 512
    cam_c = 256
    cam_f = 512
    focal_length=-2
    depth_scale = 100
    dataset_root = "/mnt/NVMe-2TB/qiaojun/SensatUrban"
    dataset_idx = "birmingham_2"
    img_name = "0223"

    # 1. Load the data. Include the RGB image, the sparse depth and the dense depth (for evaluation only)
    rgb_img = imread(os.path.join(dataset_root,dataset_idx,"Images",img_name+".png"))
    depth_img = imread(os.path.join(dataset_root,dataset_idx,"Depths",img_name+".png"))/depth_scale
    sparse_depth_img = imread(os.path.join(dataset_root,dataset_idx,"Depth_sparse",img_name+".png"))/depth_scale
    sparse_depth_mask = sparse_depth_img>0
    sparse_depth_img_gt = depth_img*sparse_depth_mask
    num_sparse_depth = np.sum(sparse_depth_mask)
    error_sparse_depth = np.sum(np.abs(depth_img-sparse_depth_img)*sparse_depth_mask)/num_sparse_depth
    error_bar = np.abs(sparse_depth_img_gt-sparse_depth_img).ravel()
    mean_depth = np.sum(sparse_depth_img)/num_sparse_depth
    mean_depth_gt = np.sum(sparse_depth_img_gt)/num_sparse_depth

    print("mean_depth",mean_depth,"mean_depth_gt",mean_depth_gt)
    print("error_sparse_depth",error_sparse_depth)
    print("num_sparse_depth",num_sparse_depth)

    # 2. Generate the mesh (vertices+faces) and initialize with the mean of the sparse depth.
    t_start = time.time()
    #vertices,faces,Laplacian = regular_512_576()
    vertices,faces,Laplacian = regular_512_1024()
    vertices = (vertices-cam_c)/cam_f
    vertices = np.hstack((vertices,np.ones((vertices.shape[0],1))))
    #vertices *= mean_depth
    print("Generate Mesh",time.time()-t_start)

    # 3. Use PyTorch3D to optimize the mesh with sparse supervision
    # function to render a mesh
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    expected_verts = torch.tensor(vertices,dtype=torch.float32,device=device)
    expected_faces = torch.tensor(faces,dtype=torch.int32,device=device)
    mesh = Meshes(verts=[expected_verts], faces=[expected_faces])
    mesh.to(device=device)
    print("Mesh to Pytorch3D",time.time()-t_start)

    cameras = SfMPerspectiveCameras(device=device, focal_length=focal_length,)
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0001,
        faces_per_pixel=1, 
    )
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    
    renderer = MeshRendererWithFragmentsOnly(rasterizer=rasterizer)
    fragments = renderer(mesh)
    depth = fragments.zbuf[0, ..., 0].detach().cpu().numpy()
    pix_to_face = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy()
    bary_coords = fragments.bary_coords[0, ..., 0,:].detach().cpu().numpy()
    print("Mesh Rendering",time.time()-t_start)

    t_start = time.time()
    A = np.zeros([num_sparse_depth+1024,1024])
    b = np.zeros(num_sparse_depth+1024)
    sparse_depth_idx = np.where(sparse_depth_mask==1)
    print(sparse_depth_idx[0].shape)
    for i in range(num_sparse_depth):
        d_idx = [sparse_depth_idx[0][i],sparse_depth_idx[1][i]]
        face_idx = pix_to_face[d_idx[0],d_idx[1]]
        v_idx = faces[face_idx]
        bary_c = bary_coords[d_idx[0],d_idx[1],:]
        A[i,v_idx] = bary_c
        b[i] = sparse_depth_img[d_idx[0],d_idx[1]]
    A[-1024:,:] = Laplacian*0.5

    '''
    x = cp.Variable(1024)
    cost = cp.sum_squares(A @ x - b)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve() 
    # Print result.
    #print("\nThe optimal value is", prob.value)
    #print("The optimal x is")
    #print(x.value)
    p = x.value
    print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)
    '''
    p, _ = torch.lstsq(torch.tensor(b).unsqueeze(1),torch.tensor(A))

    #new_vertices = vertices
    #new_vertices[:,2] = p
    #new_vertices[:,2] = p.numpy()[:1024,0]
    new_vertices = vertices * p.numpy()[:1024,:]

    print(time.time()-t_start)

    

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    #o3d_mesh.paint_uniform_color([0,0,1])

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=512, height=512, fx=512, fy=512, cx=256, cy=256)
    pcd_sparse = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(sparse_depth_img.astype(np.float32)),intrinsic=intrinsic,depth_scale=1,depth_trunc=500)
    #pcd_sparse.paint_uniform_color([1,0,0])

    o3d.visualization.draw_geometries([o3d_mesh,pcd_sparse],mesh_show_wireframe=True,mesh_show_back_face=True)
