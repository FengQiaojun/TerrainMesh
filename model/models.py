import torch
import torch.nn as nn
from collections import OrderedDict
from pytorch3d.ops import vert_align
from pytorch3d.renderer import TexturesVertex
from .deeplab import deeplabv3_resnet50, deeplabv3_resnet34, deeplabv3_resnet18
from .image_backbone import build_backbone
from .mesh_head import MeshRefinementHead
from utils.project_verts import project_verts

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

<<<<<<< HEAD
            if cfg.MODEL.BACKBONE == "resnet50":
                self.sem_model = deeplabv3_resnet50(cfg)
            elif cfg.MODEL.BACKBONE == "resnet34":
                self.sem_model = deeplabv3_resnet34(cfg)
            elif cfg.MODEL.BACKBONE == "resnet18":
                self.sem_model = deeplabv3_resnet18(cfg)
            
            if len(cfg.MODEL.MESH_HEAD.SEM_PRETRAIN_MODEL_PATH) > 0:
                checkpoint = torch.load(cfg.MODEL.MESH_HEAD.SEM_PRETRAIN_MODEL_PATH)
                model_param_names = list(checkpoint["model_state_dict"].keys())
                if "module" in model_param_names[0]:
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint["model_state_dict"].items():
                        name = k.replace("module.", "") # remove `module.`
                        new_state_dict[name] = v
                    self.sem_model.load_state_dict(new_state_dict)
                else:
                    self.sem_model.load_state_dict(checkpoint["model_state_dict"])
                if cfg.MODEL.MESH_HEAD.FREEZE_CLASSIFIER:
                    for param in self.sem_model.classifier.parameters():
                        param.requires_grad = False 
            self.backbone, feat_dims = build_backbone(cfg.MODEL.BACKBONE,in_channels=cfg.MODEL.CHANNELS,ref_model=self.sem_model.backbone,pretrained=False)
        else:
            self.backbone, feat_dims = build_backbone(cfg.MODEL.BACKBONE,in_channels=cfg.MODEL.CHANNELS,ref_model=None,pretrained=False)
=======
        self.backbone, feat_dims = build_backbone(cfg.MODEL.BACKBONE,in_channels=cfg.MODEL.CHANNELS,ref_model=None,pretrained=False)
>>>>>>> new_semantic
        
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

<<<<<<< HEAD
        refined_meshes = self.mesh_head(img_feats, init_meshes, P)

        return refined_meshes
=======
>>>>>>> new_semantic
