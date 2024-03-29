import torch
import torch.nn as nn
from collections import OrderedDict
from pytorch3d.ops import vert_align
from pytorch3d.renderer import TexturesVertex
from .deeplab import deeplabv3_resnet50, deeplabv3_resnet34, deeplabv3_resnet18
from .image_backbone import build_backbone
from .mesh_head import MeshRefinementHead

_FEAT_DIMS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
    "resnet152": (256, 512, 1024, 2048),
}

class VoxMeshHead(nn.Module):
    def __init__(self, cfg):
        super(VoxMeshHead, self).__init__()

        self.in_channels = cfg.MODEL.CHANNELS
        self.semantic = cfg.MODEL.SEMANTIC

        self.num_vertices = cfg.MODEL.MESH_HEAD.NUM_VERTICES 
        self.num_classes = cfg.MODEL.MESH_HEAD.NUM_CLASSES

        self.backbone, feat_dims = build_backbone(cfg.MODEL.BACKBONE,in_channels=cfg.MODEL.CHANNELS,ref_model=None,pretrained=False)
        
        cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS = sum(feat_dims)
        self.mesh_head = MeshRefinementHead(cfg)
        focal_length = cfg.MODEL.MESH_HEAD.FOCAL_LENGTH
        K = [
                [focal_length, 0.0, 0.0, 0.0],
                [0.0, focal_length, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        self.K = torch.tensor(K)
        
    def _get_projection_matrix(self, N, device):
        return self.K[None].repeat(N, 1, 1).to(device).detach()

    def forward(self, imgs, init_meshes, sem_2d, return_init=False):
        N = imgs.shape[0]
        device = imgs.device
        img_feats = self.backbone(imgs)
        P = self._get_projection_matrix(N, device)

        # init mesh vertex semantic features
        if self.semantic:
            vert_pos_padded = project_verts(init_meshes.verts_padded(), P)
            vert_align_feats = vert_align(sem_2d, vert_pos_padded)
            init_meshes.textures = TexturesVertex(verts_features=vert_align_feats)
        else:
            init_meshes.textures = TexturesVertex(verts_features=torch.zeros((N, self.num_vertices, self.num_classes), device=device))
        
        refined_meshes = self.mesh_head(img_feats, init_meshes, sem_2d, P)
        

        if return_init:
            return refined_meshes, init_meshes
        else:
            return refined_meshes

def project_verts(verts, P, eps=1e-1):
    """
    Project verticies using a 4x4 transformation matrix
    Inputs:
    - verts: FloatTensor of shape (N, V, 3) giving a batch of vertex positions.
    - P: FloatTensor of shape (N, 4, 4) giving projection matrices
    Outputs:
    - verts_out: FloatTensor of shape (N, V, 3) giving vertex positions (x, y, z)
        where verts_out[i] is the result of transforming verts[i] by P[i].
    """
    # Handle unbatched inputs
    singleton = False
    if verts.dim() == 2:
        assert P.dim() == 2
        singleton = True
        verts, P = verts[None], P[None]

    N, V = verts.shape[0], verts.shape[1]
    dtype, device = verts.dtype, verts.device

    # Add an extra row of ones to the world-space coordinates of verts before
    # multiplying by the projection matrix. We could avoid this allocation by
    # instead multiplying by a 4x3 submatrix of the projectio matrix, then
    # adding the remaining 4x1 vector. Not sure whether there will be much
    # performance difference between the two.
    ones = torch.ones(N, V, 1, dtype=dtype, device=device)
    verts_hom = torch.cat([verts, ones], dim=2)
    verts_cam_hom = torch.bmm(verts_hom, P.transpose(1, 2))

    # Avoid division by zero by clamping the absolute value
    w = verts_cam_hom[:, :, 2:3]
    w_sign = w.sign()
    w_sign[w == 0] = 1
    w = w_sign * w.abs().clamp(min=eps)

    verts_proj = verts_cam_hom[:, :, :3] / w

    if singleton:
        return verts_proj[0]
    return verts_proj