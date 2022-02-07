# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
)

logger = logging.getLogger(__name__)

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

class MeshHybridLoss(nn.Module):
    def __init__(
        self,
        image_size,
        focal_length,
        chamfer_weight=1.0,
        depth_weight=1.0,
        normal_weight=0.0,
        edge_weight=0.1,
        laplacian_weight=0.1,
        semantic_weight=0.1,
        gt_num_samples=10000,
        pred_num_samples=10000,
        semantic=True,
        graph_conv_semantic=True,
        device=None):
        super(MeshHybridLoss, self).__init__()
        self.image_size = image_size
        self.focal_length = focal_length
        self.depth_weight = depth_weight
        self.chamfer_weight = chamfer_weight
        self.normal_weight = normal_weight
        self.edge_weight = edge_weight
        self.laplacian_weight = laplacian_weight
        self.semantic_weight = semantic_weight
        self.gt_num_samples = gt_num_samples
        self.pred_num_samples = pred_num_samples
        self.semantic = semantic
        self.graph_conv_semantic = graph_conv_semantic
        R = torch.eye(3).reshape((1,3,3))
        R = R.to(device)
        T = torch.zeros(1,3)
        T = T.to(device)
        cameras = SfMPerspectiveCameras(device=device, R=R, T=T, focal_length=-self.focal_length)
        # Define the Rasterization
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0001, 
            faces_per_pixel=1, 
        )
        # Define the renderer
        self.renderer_depth = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
        )
        self.renderer_semantic = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader = TextureShader(device=device)
        )


    def forward(self, meshes_pred, sem_2d_pred, meshes_gt, gt_depth, gt_semantic):
        
        total_loss = torch.tensor(0.0)
        losses = {}

        # Sample from meshes_gt if we haven't already
        if isinstance(meshes_gt, Meshes):
            points_gt, normals_gt = sample_points_from_meshes(
                meshes_gt, num_samples=self.gt_num_samples, return_normals=True
            )
        else:
            points_gt = meshes_gt
            normals_gt = None

        for i, cur_meshes_pred in enumerate(meshes_pred):
            # 3D loss (Chamfer)
            if (self.chamfer_weight > 0):
                points_pred, normals_pred = sample_points_from_meshes(
                    cur_meshes_pred, num_samples=self.pred_num_samples, return_normals=True
                )
                if normals_gt!=None:
                    cham_loss, normal_loss = chamfer_distance(
                        points_pred, points_gt, x_normals=normals_pred, y_normals=normals_gt
                    )
                else:
                    cham_loss, _ = chamfer_distance(
                        points_pred, points_gt
                    )
                    normal_loss = 0
                total_loss = total_loss + self.chamfer_weight * cham_loss
                total_loss = total_loss + self.normal_weight * normal_loss
                losses["chamfer_%d"%i] = cham_loss
                losses["normal_%d"%i] = normal_loss
            # 2D loss (render)
            if (self.depth_weight > 0):
                depth = self.renderer_depth(cur_meshes_pred)
                depth = depth.permute(0,3,1,2)
                depth_available_map = (depth>0)*(gt_depth>0)
                loss_depth = torch.sum(torch.abs(gt_depth-depth)*depth_available_map,dim=(1,2,3))/torch.sum(depth_available_map,dim=(1,2,3))
                depth_loss = torch.mean(loss_depth)
                total_loss = total_loss + self.depth_weight * depth_loss
                losses["depth_%d"%i] = depth_loss
                losses["coverage_%d"%i] = torch.mean(torch.true_divide(torch.sum(depth>0,dim=(1,2,3)), 512**2))
            if (self.edge_weight > 0):
                edge_loss = mesh_edge_loss(cur_meshes_pred)
                total_loss = total_loss + self.edge_weight * edge_loss
                losses["edge_%d"%i] = edge_loss
            if (self.laplacian_weight > 0):
                laplacian_loss = mesh_laplacian_smoothing(cur_meshes_pred)
                total_loss = total_loss + self.laplacian_weight * laplacian_loss
                losses["laplacian_%d"%i] = laplacian_loss
            # Semantic Segmentation weight
            if (self.semantic and self.graph_conv_semantic and self.semantic_weight > 0):
                semantic_predict = self.renderer_semantic(cur_meshes_pred).permute(0,3,1,2)
                criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
                semantic_loss = criterion(semantic_predict, gt_semantic)
                total_loss = total_loss + self.semantic_weight * semantic_loss
                losses["semantic_%d"%i] = semantic_loss

        if sem_2d_pred is not None:
            criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
            semantic_2d_loss = criterion(sem_2d_pred, gt_semantic)
            total_loss = total_loss + self.semantic_weight * semantic_2d_loss
            losses["semantic"] = semantic_2d_loss

        return total_loss, losses


class FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=1, gamma=0, size_average=True, ignore_index=255,):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
                inputs, targets, weight = self.weight, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


