# Test how far we can go if we optimize a single mesh.
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from imageio import imwrite

import torch
import torch.nn as nn
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
lr = 1e-2
iters = 300

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        
    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        return fragments.zbuf

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


if __name__ == "__main__":

    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)

    # Specify the GPU
    worker_id = cfg.SOLVER.GPU_ID
    device = torch.device("cuda:%d" % worker_id)

    # Build the loss
    loss_fn_kwargs = {
        "chamfer_weight": cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT,
        "depth_weight": cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT,
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

    rgb_img, sparse_depth, depth_edt, sem_pred, mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, gt_semantic = load_data_by_index(cfg = cfg, seq_idx = "birmingham_5",img_idx="0012",meshing="mesh1024",samples="1000",device=device)
    
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
    )
    # Define the renderer
    renderer_depth = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
    )
    renderer_semantic = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader = TextureShader(device=device)
            #shader = SoftPhongShader(device=device, cameras=cameras)
    )

    focal_length = cfg.MODEL.MESH_HEAD.FOCAL_LENGTH
    K = [
                [focal_length, 0.0, 0.0, 0.0],
                [0.0, focal_length, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
    K = torch.tensor(K)
    P = K[None].repeat(1, 1, 1).to(device).detach()
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
    #optimizer = torch.optim.Adam([deform_verts, sem_verts], lr=lr)

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
        #new_src_mesh = mesh.offset_verts(deform_verts)
        #new_src_mesh.texture = TexturesVertex(verts_features=vert_align_feats+sem_verts.view(1,1024,5))
        #new_src_mesh.textures._verts_features_padded = mesh.textures.verts_features_padded()+sem_verts.view(1,1024,5)
        #new_src_mesh.texture = TexturesVertex(verts_features=sem_verts.view(1,1024,5))
        

        loss, losses = loss_fn([new_src_mesh], None, gt_mesh_pcd, gt_depth, gt_semantic)
        loop.set_description('total_loss = %.6f, chamfer_loss = %.6f, depth_loss = %.6f, sem_loss = %.6f.' % (loss, losses["chamfer_0"], losses["depth_0"], losses["semantic_0"]))
        #loop.set_description('total_loss = %.6f, chamfer_loss = %.6f, depth_loss = %.6f.' % (loss, losses["chamfer_0"], losses["depth_0"]))
        # Optimization step
        loss.backward()
        optimizer.step()    
        #print(deform_verts.grad)
        #print(sem_verts.grad)

        depth = renderer_depth(new_src_mesh).permute(0,3,1,2).detach().cpu().numpy()
        semantic_predict = renderer_semantic(new_src_mesh).argmax(3)[0,::].detach().cpu().numpy()
        
        if i%10 == 0:
            imwrite("visualizations/%d.png"%i,convert_class_to_rgb_sensat_simplified(semantic_predict))
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