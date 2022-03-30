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
from imageio import imwrite
from utils.semantic_labels import convert_class_to_rgb_sensat_simplified

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
        class_weight=None,
        sem_loss_func=None,
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
        self.class_weight = class_weight
        self.sem_loss_func = sem_loss_func
        self.device = device
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
            perspective_correct=False, # this seems solve the nan error
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
        if self.sem_loss_func == "CrossEntropy":
            self.sem_criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device),ignore_index=0, reduction='mean')
        elif self.sem_loss_func == "Focal":
            self.sem_criterion = FocalLoss(weight=torch.Tensor(self.class_weight).to(self.device), ignore_index=0, size_average=True)
        elif self.sem_loss_func == "Dice":
            self.sem_criterion = DiceLoss(num_classes=5,weights=torch.Tensor(self.class_weight).to(self.device),device=self.device,eps=1e-7)
        elif self.sem_loss_func == "Jaccard":
            self.sem_criterion = JaccardLoss(num_classes=5,weights=torch.Tensor(self.class_weight).to(self.device),device=self.device,eps=1e-7)
        else:
            self.sem_criterion = None         


    def set_semantic_weight(self, semantic_weight):
        self.semantic_weight = semantic_weight

    def get_semantic_weight(self):
        return self.semantic_weight

    def forward(self, meshes_pred, sem_2d_pred, meshes_gt, gt_depth, gt_semantic, return_img=False):
        
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
            if (points_gt is not None and self.chamfer_weight > 0):
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
            if (gt_depth is not None and self.depth_weight > 0):
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
            # Semantic Laplacian
            if (self.laplacian_weight > 0 and gt_semantic is not None and self.semantic_weight > 0):
                laplacian_sem_loss = mesh_laplacian_smoothing_texture(cur_meshes_pred)
                total_loss = total_loss + self.laplacian_weight * laplacian_sem_loss
                losses["laplacian_sem_%d"%i] = laplacian_sem_loss
            # Semantic Segmentation weight
            if (gt_semantic is not None and self.semantic_weight > 0):
                semantic_predict = self.renderer_semantic(cur_meshes_pred).permute(0,3,1,2)
                semantic_loss = self.sem_criterion(semantic_predict, gt_semantic)
                total_loss = total_loss + self.semantic_weight * semantic_loss
                losses["semantic_%d"%i] = semantic_loss
        
        if return_img:
            img_list = []
            if (gt_depth is not None and self.depth_weight > 0):
                img_list.append(depth)
            if (gt_semantic is not None and self.semantic_weight > 0):
                img_list.append(semantic_predict)
            return total_loss, losses, img_list
        else:
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


class DiceLoss(nn.Module):
    def __init__(self, num_classes, weights, device, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.weights = weights
        self.device = device
        self.eps = eps
    
    def forward(self, inputs, targets):
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            inputs: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            targets: a tensor of shape [B, 1, H, W].
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        num_classes = self.num_classes
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[targets.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(inputs)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes,device=self.device)[targets.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(inputs, dim=1)
        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps))
        dice_loss = torch.dot(dice_loss.mean(1), self.weights)/torch.sum(self.weights)
        return (1 - dice_loss)

class JaccardLoss(nn.Module):
    def __init__(self, num_classes, weights, device, eps=1e-7):
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.weights = weights
        self.device = device
        self.eps = eps

    def forward(self, inputs, targets):
        """Computes the Jaccard loss, a.k.a the IoU loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the jaccard loss so we
        return the negated jaccard loss.
        Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            jacc_loss: the Jaccard loss.
        """
        if self.num_classes == 1:
            true_1_hot = torch.eye(self.num_classes + 1)[targets.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(inputs)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(self.num_classes,device=self.device)[targets.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(inputs, dim=1)
        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + self.eps))
        jacc_loss = torch.dot(jacc_loss.mean(1), self.weights)/torch.sum(self.weights)
        return (1 - jacc_loss)

def mesh_laplacian_smoothing_texture(meshes, method: str = "uniform"):
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )
    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    tex_packed = meshes.textures.verts_features_packed()
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()
    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        else:
            raise ValueError("Method should be one of {uniform}")
    if method == "uniform":
        loss = L.mm(tex_packed)
    loss = loss.norm(dim=1)
    loss = loss * weights
    return loss.sum() / N