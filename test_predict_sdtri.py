import numpy as np 
from scipy.spatial import Delaunay
import open3d as o3d
import matplotlib.pyplot as plt
from imageio import imread 
import os 
from tqdm import tqdm

import torch 
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj,load_objs_as_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing

from config import get_sensat_cfg
from dataset.sensat_dataset import load_data_by_index
from loss import MeshHybridLoss

def triangulation_sfm_points(vertices):
    vertices_2D = vertices[:,:2]/np.expand_dims(vertices[:,2],axis=1)
    tri = Delaunay(vertices_2D)
    faces = tri.simplices
    return vertices,faces

def triangulation_sfm_512(sparse_depth,image_size=512,focal_length=2):
    cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=image_size,height=image_size,fx=image_size/2*focal_length,fy=image_size/2*focal_length,cx=image_size/2,cy=image_size/2)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(sparse_depth.astype(np.float32)), intrinsic=cam_intrinsic)
    #plt.imshow(sparse_depth)
    #plt.show()
    #print(pcd)
    #o3d.visualization.draw_geometries([pcd])
    points = np.array(pcd.points)
    vertices,faces = triangulation_sfm_points(points)
    return vertices,faces

cfg_file = "Sensat_predict.yaml"
lr = 1e-2
iters = 30

if __name__ == "__main__":

    # Load the config and create a folder to save the outputs.
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
        "semantic_weight": 0,
        "gt_num_samples": cfg.MODEL.MESH_HEAD.GT_NUM_SAMPLES,
        "pred_num_samples": cfg.MODEL.MESH_HEAD.PRED_NUM_SAMPLES,
        "image_size": cfg.MODEL.MESH_HEAD.IMAGE_SIZE,
        "focal_length": cfg.MODEL.MESH_HEAD.FOCAL_LENGTH,
        "semantic": False,
        "class_weight": cfg.MODEL.DEEPLAB.CLASS_WEIGHT,
        "sem_loss_func": cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_FUNC,
        "device": device
    }
    loss_fn = MeshHybridLoss(**loss_fn_kwargs)

    num_count = 0
    loss_sum = 0
    loss_chamfer_sum = 0
    loss_depth_sum = 0

    for seq_idx in ["birmingham_4","cambridge_10","cambridge_11"]:
        for img_idx in range(660):
            img_idx = "%04d"%img_idx 
            rgb_img, sparse_depth, depth_edt, sem_2d_pred, init_mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, gt_semantic = load_data_by_index(cfg = cfg, seq_idx = seq_idx,img_idx=img_idx,meshing="mesh1024",samples="1000",device=device)
                
            #sparse_depth = sparse_depth.cpu().numpy()[0,0,::]
            sparse_depth_path = os.path.join("/mnt/NVMe-2TB/qiaojun/SensatUrban",seq_idx,"Pcds_500",img_idx+".png")
            sparse_depth = np.asfarray(imread(sparse_depth_path)/100, dtype=np.float32)
            vertices, faces = triangulation_sfm_512(sparse_depth)
            torch_verts = torch.tensor(vertices,dtype=torch.float32)
            torch_faces = torch.tensor(faces,dtype=torch.int64)
            #tri_mesh = Meshes(verts=list(torch_verts),faces=list(torch_faces),)        
            save_obj("temp.obj", torch_verts, torch_faces)       
            tri_mesh = load_objs_as_meshes(["temp.obj"], device=device)
            
            loss, losses, img_predict = loss_fn(tri_mesh, None, gt_mesh_pcd, gt_depth, gt_semantic, return_img=True)
            print("loss_chamfer[0]",losses["chamfer_0"])
            print("loss_depth[0]",losses["depth_0"])
            '''
            # Perform some smoothing
            verts, faces = tri_mesh.get_mesh_verts_faces(0)
            deform_verts = torch.full(
                            tri_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([deform_verts], lr=lr)
            loop = tqdm(range(iters), disable=False)
            for i in loop:
                # Initialize optimizer
                optimizer.zero_grad()
                # Deform the mesh
                new_mesh = Meshes(
                    verts=[verts.to(device)+deform_verts],   
                    faces=[faces.to(device)], 
                )
                loss = 0.5*mesh_laplacian_smoothing(new_mesh) + 0.01*mesh_edge_loss(new_mesh)
                loss.backward(retain_graph=True)
                optimizer.step()
            tri_mesh = new_mesh

            loss, losses, img_predict = loss_fn(tri_mesh, None, gt_mesh_pcd, gt_depth, gt_semantic, return_img=True)
            print("loss_chamfer[0]",losses["chamfer_0"])
            print("loss_depth[0]",losses["depth_0"])
            '''
            if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
                loss_chamfer_sum += losses["chamfer_0"].detach().cpu().numpy()
            if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
                loss_depth_sum += losses["depth_0"].detach().cpu().numpy()

            num_count += 1
            

    print(num_count)

    if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
        print("loss_chamfer_sum", loss_chamfer_sum/num_count)
    if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
        print("loss_depth_sum", loss_depth_sum/num_count)