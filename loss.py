# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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
    MeshRasterizer
)

logger = logging.getLogger(__name__)


class MeshLoss(nn.Module):
    def __init__(
        self,
        chamfer_weight=1.0,
        normal_weight=0.0,
        edge_weight=0.1,
        laplacian_weight=0.1,
        gt_num_samples=10000,
        pred_num_samples=10000,
    ):

        super(MeshLoss, self).__init__()
        self.chamfer_weight = chamfer_weight
        self.normal_weight = normal_weight
        self.edge_weight = edge_weight
        self.laplacian_weight = laplacian_weight
        self.gt_num_samples = gt_num_samples
        self.pred_num_samples = pred_num_samples
        

    def forward(self, meshes_pred, meshes_gt):
        """
        Args:
          meshes_pred: Meshes
          meshes_gt: Either Meshes, or a tuple (points_gt, normals_gt)

        Returns:
          loss (float): Torch scalar giving the total loss, or None if an error occured and
                we should skip this loss. TODO use an exception instead?
          losses (dict): A dictionary mapping loss names to Torch scalars giving their
                        (unweighted) values.
        """
        # Sample from meshes_gt if we haven't already
        if isinstance(meshes_gt, Meshes):
            points_gt, normals_gt = sample_points_from_meshes(
                meshes_gt, num_samples=self.gt_num_samples, return_normals=True
            )
        else:
            points_gt = meshes_gt
            normals_gt = None

        total_loss = torch.tensor(0.0).to(points_gt)
        losses = {}

        if isinstance(meshes_pred, Meshes):
            meshes_pred = [meshes_pred]
        elif meshes_pred is None:
            meshes_pred = []

        for i, cur_meshes_pred in enumerate(meshes_pred):
            cur_out = self._mesh_loss(cur_meshes_pred, points_gt, normals_gt)
            cur_loss, cur_losses = cur_out
            if total_loss is None or cur_loss is None:
                total_loss = None
            else:
                total_loss = total_loss + cur_loss #/ len(meshes_pred)
            for k, v in cur_losses.items():
                losses["%s_%d" % (k, i)] = v

        return total_loss, losses

    def _mesh_loss(self, meshes_pred, points_gt, normals_gt=None):
        """
        Args:
          meshes_pred: Meshes containing N meshes
          points_gt: Tensor of shape NxPx3
          normals_gt: Tensor of shape NxPx3

        Returns:
          total_loss (float): The sum of all losses specific to meshes
          losses (dict): All (unweighted) mesh losses in a dictionary
        """
        zero = torch.tensor(0.0).to(meshes_pred.verts_list()[0])
        losses = {"chamfer": zero, "normal": zero, "edge": zero}
        points_pred, normals_pred = sample_points_from_meshes(
            meshes_pred, num_samples=self.pred_num_samples, return_normals=True
        )

        total_loss = torch.tensor(0.0).to(points_pred)
        
        losses = {}
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
        losses["chamfer"] = cham_loss
        losses["normal"] = normal_loss

        edge_loss = mesh_edge_loss(meshes_pred)
        total_loss = total_loss + self.edge_weight * edge_loss
        losses["edge"] = edge_loss

        laplacian_loss = mesh_laplacian_smoothing(meshes_pred)
        total_loss = total_loss + self.laplacian_weight * laplacian_loss
        losses["laplacian"] = laplacian_loss

        return total_loss, losses



class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        
    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        return fragments.zbuf

class MeshRenderLoss(nn.Module):
    def __init__(
        self,
        depth_weight=1.0,
        edge_weight=0.1,
        laplacian_weight=0.1,
        device=None):
        super(MeshRenderLoss, self).__init__()
        R = torch.eye(3).reshape((1,3,3))
        R = R.to(device)
        T = torch.zeros(1,3)
        T = T.to(device)
        cameras = SfMPerspectiveCameras(device=device, R=R, T=T, focal_length=-10)
        # Define the Rasterization
        raster_settings = RasterizationSettings(
            image_size=512, 
            blur_radius=0.0001, 
            faces_per_pixel=1, 
        )
        # Define the renderer
        self.renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
        )
        self.depth_weight = depth_weight
        self.edge_weight = edge_weight
        self.laplacian_weight = laplacian_weight

    def forward(self, meshes_pred, gt_depth):
        
        total_loss = torch.tensor(0.0)
        losses = {}

        for i, cur_meshes_pred in enumerate(meshes_pred):
            depth = self.renderer(cur_meshes_pred)
            depth = depth.permute(0,3,1,2)
            depth_available_map = (depth>0)*(gt_depth>0)
            loss_depth = torch.sum(torch.abs(gt_depth-depth)*depth_available_map,dim=(1,2,3))/torch.sum(depth_available_map,dim=(1,2,3))
            depth_loss = torch.mean(loss_depth)
            total_loss = total_loss + self.depth_weight * depth_loss
            losses["depth_%d"%i] = depth_loss
            losses["coverage_%d"%i] = torch.mean(torch.true_divide(torch.sum(depth>0,dim=(1,2,3)), 512**2))

            edge_loss = mesh_edge_loss(cur_meshes_pred)
            total_loss = total_loss + self.edge_weight * edge_loss
            losses["edge_%d"%i] = edge_loss

            laplacian_loss = mesh_laplacian_smoothing(cur_meshes_pred)
            total_loss = total_loss + self.laplacian_weight * laplacian_loss
            losses["laplacian_%d"%i] = laplacian_loss
        
        return total_loss, losses



class MeshHybridLoss(nn.Module):
    def __init__(
        self,
        chamfer_weight=1.0,
        depth_weight=1.0,
        normal_weight=0.0,
        edge_weight=0.1,
        laplacian_weight=0.1,
        sem_weight=0.1,
        gt_num_samples=10000,
        pred_num_samples=10000,
        device=None):
        super(MeshHybridLoss, self).__init__()
        self.depth_weight = depth_weight
        self.chamfer_weight = chamfer_weight
        self.normal_weight = normal_weight
        self.edge_weight = edge_weight
        self.laplacian_weight = laplacian_weight
        self.gt_num_samples = gt_num_samples
        self.pred_num_samples = pred_num_samples
        R = torch.eye(3).reshape((1,3,3))
        R = R.to(device)
        T = torch.zeros(1,3)
        T = T.to(device)
        cameras = SfMPerspectiveCameras(device=device, R=R, T=T, focal_length=-10)
        # Define the Rasterization
        raster_settings = RasterizationSettings(
            image_size=512, 
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


    def forward(self, meshes_pred, meshes_gt, gt_depth, gt_sem):
        
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
            if (self.sem_weight > 0):
                # TODO
                sem_loss = 

        return total_loss, losses