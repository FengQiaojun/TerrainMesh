# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from pytorch3d.ops import GraphConv, SubdivideMeshes, vert_align
from pytorch3d.renderer import TexturesVertex
from torch.nn import functional as F

from utils.project_verts import project_verts


class MeshRefinementHead(nn.Module):
    def __init__(self, cfg):
        super(MeshRefinementHead, self).__init__()

        # fmt: off
        semantic        = cfg.MODEL.MESH_HEAD.GRAPH_CONV_SEMANTIC
        dim_semantic    = cfg.MODEL.MESH_HEAD.GRAPH_CONV_DIM_SEMANTIC
        input_channels  = cfg.MODEL.MESH_HEAD.COMPUTED_INPUT_CHANNELS
        self.num_stages = cfg.MODEL.MESH_HEAD.NUM_STAGES
        self.stage_list = cfg.MODEL.MESH_HEAD.STAGE_LIST
        hidden_dim      = cfg.MODEL.MESH_HEAD.GRAPH_CONV_DIM
        stage_depth     = cfg.MODEL.MESH_HEAD.NUM_GRAPH_CONVS
        graph_conv_init = cfg.MODEL.MESH_HEAD.GRAPH_CONV_INIT
        num_classes     = cfg.MODEL.MESH_HEAD.NUM_CLASSES
        num_vertices    = cfg.MODEL.MESH_HEAD.NUM_VERTICES
        sem_residual    = cfg.MODEL.MESH_HEAD.SEMANTIC_RESIDUAL
        #vert_offset_threshold = cfg.MODEL.MESH_HEAD.OFFSET_THRESHOLD
        vert_offset_threshold = None
        # fmt: on

        self.stages = nn.ModuleList()
        if len(self.stage_list) != self.num_stages:
            print("Error in cfg.MODEL.MESH_HEAD.STAGE_LIST: number does not agree.")
        for i in range(self.num_stages):
            vert_feat_dim = 0 if i == 0 else hidden_dim
            if self.stage_list[i] == "geo":
                stage = MeshGeoRefinementStage(
                    input_channels, vert_feat_dim, hidden_dim, num_vertices, num_classes, stage_depth, gconv_init=graph_conv_init, vert_offset_threshold=vert_offset_threshold, sem_residual=sem_residual
                )
            elif self.stage_list[i] == "sem":
                stage = MeshSemRefinementStage(
                    input_channels, vert_feat_dim, hidden_dim, num_vertices, num_classes, stage_depth, gconv_init=graph_conv_init, vert_offset_threshold=vert_offset_threshold, sem_residual=sem_residual
                )  
            elif self.stage_list[i] == "hybrid":              
                stage = MeshRefinementStage(
                    input_channels, vert_feat_dim, hidden_dim, num_vertices, num_classes, stage_depth, gconv_init=graph_conv_init, vert_offset_threshold=vert_offset_threshold, sem_residual=sem_residual
                )
            else:
                print("Error in stage name!")
            self.stages.append(stage)

    def forward(self, img_feats, meshes, sem_2d, P=None, subdivide=False):
        """
        Args:
          img_feats (tensor): Tensor of shape (N, C, H, W) giving image features,
                              or a list of such tensors.
          meshes (Meshes): Meshes class of N meshes
          P (tensor): Tensor of shape (N, 4, 4) giving projection matrix to be applied
                      to vertex positions before vert-align. If None, don't project verts.
          subdivide (bool): Flag whether to subdivice the mesh after refinement

        Returns:
          output_meshes (list of Meshes): A list with S Meshes, where S is the
                                          number of refinement stages
        """
        output_meshes = []
        vert_feats = None
        vert_sem_feats = None
        for i, stage in enumerate(self.stages):
            meshes, vert_feats, vert_sem_feats = stage(img_feats, meshes, vert_feats, vert_sem_feats, sem_2d, P)
            output_meshes.append(meshes)
            #if subdivide and i < self.num_stages - 1:
            #    subdivide = SubdivideMeshes()
            #    meshes, vert_feats = subdivide(meshes, feats=vert_feats)
        return output_meshes




class MeshGeoRefinementStage(nn.Module):
    def __init__(self, img_feat_dim, vert_feat_dim, hidden_dim, num_vertices, num_classes, stage_depth, gconv_init="normal", vert_offset_threshold=None, sem_residual=True):
        """
        Args:
          img_feat_dim (int): Dimension of features we will get from vert_align
          vert_feat_dim (int): Dimension of vert_feats we will receive from the
                               previous stage; can be 0
          hidden_dim (int): Output dimension for graph-conv layers
          stage_depth (int): Number of graph-conv layers to use
          gconv_init (int): Specifies weight initialization for graph-conv layers
        """
        super(MeshGeoRefinementStage, self).__init__()

        self.vert_offset_threshold = vert_offset_threshold
        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)
        self.vert_offset = nn.Linear(hidden_dim + 3, 3)
        
        self.gconvs = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3
            else:
                input_dim = hidden_dim + 3
            gconv = GraphConv(input_dim, hidden_dim, init=gconv_init, directed=False)
            self.gconvs.append(gconv)

        # initialization for bottleneck and vert_offset
        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)
        nn.init.zeros_(self.vert_offset.weight)
        nn.init.constant_(self.vert_offset.bias, 0)

    def forward(self, img_feats, meshes, vert_feats=None, vert_sem_feats=None, sem_2d=None, P=None):
        """
        Args:
          img_feats (tensor): Features from the backbone
          meshes (Meshes): Initial meshes which will get refined
          vert_feats (tensor): Features from the previous refinement stage
          P (tensor): Tensor of shape (N, 4, 4) giving projection matrix to be applied
                      to vertex positions before vert-align. If None, don't project verts.
        """
        # Project verts if we are making predictions in world space
        verts_padded_to_packed_idx = meshes.verts_padded_to_packed_idx()

        if P is not None:
            vert_pos_padded = project_verts(meshes.verts_padded(), P)
            #vert_pos_packed = _padded_to_packed(vert_pos_padded, verts_padded_to_packed_idx)
            vert_pos_packed = _padded_to_packed(meshes.verts_padded(), verts_padded_to_packed_idx)
        else:
            vert_pos_padded = meshes.verts_padded()
            vert_pos_packed = meshes.verts_packed()

        device, dtype = vert_pos_padded.device, vert_pos_padded.dtype
        # Get features from the image
        vert_align_feats_project = vert_align(img_feats, vert_pos_padded)
        vert_align_feats_project = _padded_to_packed(vert_align_feats_project, verts_padded_to_packed_idx)
        vert_align_feats = F.relu(self.bottleneck(vert_align_feats_project))

        # Prepare features for first graph conv layer
        first_layer_feats = [vert_align_feats, vert_pos_packed]
        if vert_feats is not None:
            first_layer_feats.append(vert_feats)
        vert_feats = torch.cat(first_layer_feats, dim=1)
        
        # Run graph conv layers
        for gconv in self.gconvs:
            vert_feats_nopos = F.relu(gconv(vert_feats, meshes.edges_packed()))
            vert_feats = torch.cat([vert_feats_nopos, vert_pos_packed], dim=1)

        # Predict a new mesh by offsetting verts
        vert_offsets = self.vert_offset(vert_feats)
        # Avoid nan
        vert_offsets[torch.where(torch.isnan(vert_offsets))] = 0
        if not (self.vert_offset_threshold is None):
            vert_offsets[torch.where(vert_offsets>self.vert_offset_threshold)] = self.vert_offset_threshold
        meshes_out = meshes.offset_verts(vert_offsets)
        
        vert_sem_feats_nopos = None

        return meshes_out, vert_feats_nopos, vert_sem_feats_nopos


class MeshSemRefinementStage(nn.Module):
    def __init__(self, img_feat_dim, vert_feat_dim, hidden_dim, num_vertices, num_classes, stage_depth, gconv_init="normal", vert_offset_threshold=None, sem_residual=True):
        """
        Args:
          img_feat_dim (int): Dimension of features we will get from vert_align
          vert_feat_dim (int): Dimension of vert_feats we will receive from the
                               previous stage; can be 0
          hidden_dim (int): Output dimension for graph-conv layers
          stage_depth (int): Number of graph-conv layers to use
          gconv_init (int): Specifies weight initialization for graph-conv layers
        """
        super(MeshSemRefinementStage, self).__init__()

        self.num_vertices = num_vertices
        self.num_classes = num_classes 
        self.vert_offset_threshold = vert_offset_threshold
        self.sem_residual = sem_residual

        self.bottleneck_semantic = nn.Linear(img_feat_dim, hidden_dim)
        self.vert_semantic = nn.Linear(hidden_dim + 3 + self.num_classes, self.num_classes)

        self.gconvs_sem = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3 + self.num_classes
            else:
                input_dim = hidden_dim + 3 + self.num_classes
            gconv = GraphConv(input_dim, hidden_dim, init=gconv_init, directed=False)
            self.gconvs_sem.append(gconv)
        
        # initialization for bottleneck and vert_offset
        nn.init.normal_(self.bottleneck_semantic.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck_semantic.bias, 0)

        nn.init.zeros_(self.vert_semantic.weight)
        nn.init.constant_(self.vert_semantic.bias, 0)

    def forward(self, img_feats, meshes, vert_feats_prev=None, vert_sem_feats_prev=None, sem_2d=None, P=None):
        """
        Args:
          img_feats (tensor): Features from the backbone
          meshes (Meshes): Initial meshes which will get refined
          vert_feats (tensor): Features from the previous refinement stage
          P (tensor): Tensor of shape (N, 4, 4) giving projection matrix to be applied
                      to vertex positions before vert-align. If None, don't project verts.
        """
        # Project verts if we are making predictions in world space
        verts_padded_to_packed_idx = meshes.verts_padded_to_packed_idx()

        if P is not None:
            vert_pos_padded = project_verts(meshes.verts_padded(), P)
            #vert_pos_packed = _padded_to_packed(vert_pos_padded, verts_padded_to_packed_idx)
            vert_pos_packed = _padded_to_packed(meshes.verts_padded(), verts_padded_to_packed_idx)
        else:
            vert_pos_padded = meshes.verts_padded()
            vert_pos_packed = meshes.verts_packed()

        vert_align_feats = vert_align(sem_2d, vert_pos_padded)
        meshes.textures = TexturesVertex(verts_features=vert_align_feats)

        device, dtype = vert_pos_padded.device, vert_pos_padded.dtype
        # Get features from the image
        vert_align_feats_project = vert_align(img_feats, vert_pos_padded)
        vert_align_feats_project = _padded_to_packed(vert_align_feats_project, verts_padded_to_packed_idx)
        
        vert_align_feats_semantic = F.relu(self.bottleneck_semantic(vert_align_feats_project))
        # Prepare features for first graph conv layer
        first_layer_feats = [vert_align_feats_semantic, vert_pos_packed, meshes.textures.verts_features_packed()]
        if vert_feats_prev is not None:
            first_layer_feats.append(vert_feats_prev)
        if vert_sem_feats_prev is not None:
            first_layer_feats.append(vert_sem_feats_prev)
        vert_feats_semantic = torch.cat(first_layer_feats, dim=1)
        # Run graph conv layers
        for gconv in self.gconvs_sem:
            vert_sem_feats_nopos = F.relu(gconv(vert_feats_semantic, meshes.edges_packed()))
            vert_feats_semantic = torch.cat([vert_sem_feats_nopos, vert_pos_packed, meshes.textures.verts_features_packed()], dim=1)
        meshes_textures = self.vert_semantic(vert_feats_semantic).view(-1, self.num_vertices, self.num_classes)
        vert_pos_padded = project_verts(meshes.verts_padded(), P)
        vert_align_sem = vert_align(sem_2d, vert_pos_padded)
        if self.sem_residual:
            meshes.textures = TexturesVertex(verts_features=vert_align_sem+meshes_textures)
        else:
            meshes.textures = TexturesVertex(verts_features=meshes_textures)

        vert_feats_nopos = None

        return meshes, vert_feats_nopos, vert_sem_feats_nopos


class MeshRefinementStage(nn.Module):
    def __init__(self, img_feat_dim, vert_feat_dim, hidden_dim, num_vertices, num_classes, stage_depth, gconv_init="normal", vert_offset_threshold=None, sem_residual=True):
        """
        Args:
          img_feat_dim (int): Dimension of features we will get from vert_align
          vert_feat_dim (int): Dimension of vert_feats we will receive from the
                               previous stage; can be 0
          hidden_dim (int): Output dimension for graph-conv layers
          stage_depth (int): Number of graph-conv layers to use
          gconv_init (int): Specifies weight initialization for graph-conv layers
        """
        super(MeshRefinementStage, self).__init__()

        self.num_vertices = num_vertices
        self.num_classes = num_classes 
        self.vert_offset_threshold = vert_offset_threshold
        self.sem_residual = sem_residual

        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)
        self.bottleneck_semantic = nn.Linear(img_feat_dim, hidden_dim)
        
        self.vert_offset = nn.Linear(hidden_dim + 3 + self.num_classes, 3)
        self.vert_semantic = nn.Linear(hidden_dim + 3 + self.num_classes, self.num_classes)
        
        self.gconvs = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3 + self.num_classes
            else:
                input_dim = hidden_dim + 3 + self.num_classes
            gconv = GraphConv(input_dim, hidden_dim, init=gconv_init, directed=False)
            self.gconvs.append(gconv)

        self.gconvs_sem = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + 3 + self.num_classes
            else:
                input_dim = hidden_dim + 3 + self.num_classes
            gconv = GraphConv(input_dim, hidden_dim, init=gconv_init, directed=False)
            self.gconvs_sem.append(gconv)

        # initialization for bottleneck and vert_offset
        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)

        nn.init.zeros_(self.vert_offset.weight)
        nn.init.constant_(self.vert_offset.bias, 0)

        nn.init.normal_(self.bottleneck_semantic.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck_semantic.bias, 0)

        nn.init.zeros_(self.vert_semantic.weight)
        nn.init.constant_(self.vert_semantic.bias, 0)

    def forward(self, img_feats, meshes, vert_feats_prev=None, vert_sem_feats_prev=None, sem_2d=None, P=None):
        """
        Args:
          img_feats (tensor): Features from the backbone
          meshes (Meshes): Initial meshes which will get refined
          vert_feats_prev (tensor): Features from the previous refinement stage
          P (tensor): Tensor of shape (N, 4, 4) giving projection matrix to be applied
                      to vertex positions before vert-align. If None, don't project verts.
        """
        # Project verts if we are making predictions in world space
        verts_padded_to_packed_idx = meshes.verts_padded_to_packed_idx()

        if P is not None:
            vert_pos_padded = project_verts(meshes.verts_padded(), P)
            #vert_pos_packed = _padded_to_packed(vert_pos_padded, verts_padded_to_packed_idx)
            vert_pos_packed = _padded_to_packed(meshes.verts_padded(), verts_padded_to_packed_idx)
        else:
            vert_pos_padded = meshes.verts_padded()
            vert_pos_packed = meshes.verts_packed()
        vert_align_feats = vert_align(sem_2d, vert_pos_padded)
        meshes.textures = TexturesVertex(verts_features=vert_align_feats)

        device, dtype = vert_pos_padded.device, vert_pos_padded.dtype
        # Get features from the image
        vert_align_feats_project = vert_align(img_feats, vert_pos_padded)
        vert_align_feats_project = _padded_to_packed(vert_align_feats_project, verts_padded_to_packed_idx)
        vert_align_feats = F.relu(self.bottleneck(vert_align_feats_project))

        # Prepare features for first graph conv layer
        first_layer_feats = [vert_align_feats, vert_pos_packed, meshes.textures.verts_features_packed()]
        if vert_feats_prev is not None:
            first_layer_feats.append(vert_feats_prev)
        vert_feats = torch.cat(first_layer_feats, dim=1)
        
        # Run graph conv layers
        for gconv in self.gconvs:
            vert_feats_nopos = F.relu(gconv(vert_feats, meshes.edges_packed()))
            vert_feats = torch.cat([vert_feats_nopos, vert_pos_packed, meshes.textures.verts_features_packed()], dim=1)

        # Predict a new mesh by offsetting verts
        vert_offsets = self.vert_offset(vert_feats)
        # Avoid nan
        vert_offsets[torch.where(torch.isnan(vert_offsets))] = 0
        if not (self.vert_offset_threshold is None):
            vert_offsets[torch.where(vert_offsets>self.vert_offset_threshold)] = self.vert_offset_threshold
        meshes_out = meshes.offset_verts(vert_offsets)
        
        '''# Re-retrieve the semantic features
        if P is not None:
            vert_pos_padded = project_verts(meshes.verts_padded(), P)
            #vert_pos_packed = _padded_to_packed(vert_pos_padded, verts_padded_to_packed_idx)
            vert_pos_packed = _padded_to_packed(meshes.verts_padded(), verts_padded_to_packed_idx)
        else:
            vert_pos_padded = meshes.verts_padded()
            vert_pos_packed = meshes.verts_packed()
        vert_align_feats = vert_align(sem_2d, vert_pos_padded)
        meshes.textures = TexturesVertex(verts_features=vert_align_feats)
        vert_align_feats_project = vert_align(img_feats, vert_pos_padded)
        vert_align_feats_project = _padded_to_packed(vert_align_feats_project, verts_padded_to_packed_idx)
        '''# Re-retrieve the semantic features
        
        vert_align_feats_semantic = F.relu(self.bottleneck_semantic(vert_align_feats_project))
        # Prepare features for first graph conv layer
        first_layer_feats = [vert_align_feats_semantic, vert_pos_packed, meshes.textures.verts_features_packed()]
        if vert_feats_prev is not None:
            first_layer_feats.append(vert_feats_prev)
        if vert_sem_feats_prev is not None:
            first_layer_feats.append(vert_sem_feats_prev)
        vert_feats_semantic = torch.cat(first_layer_feats, dim=1)
            # Run graph conv layers
        for gconv in self.gconvs_sem:
            vert_sem_feats_nopos = F.relu(gconv(vert_feats_semantic, meshes.edges_packed()))
            vert_feats_semantic = torch.cat([vert_sem_feats_nopos, vert_pos_packed, meshes.textures.verts_features_packed()], dim=1)
        meshes_textures = self.vert_semantic(vert_feats_semantic).view(-1, self.num_vertices, self.num_classes)
        vert_pos_padded = project_verts(meshes_out.verts_padded(), P)
        vert_align_sem = vert_align(sem_2d, vert_pos_padded)
        if self.sem_residual:
            meshes_out.textures = TexturesVertex(verts_features=vert_align_sem+meshes_textures)
        else:
            meshes_out.textures = TexturesVertex(verts_features=meshes_textures)

        return meshes_out, vert_feats_nopos, vert_sem_feats_nopos





def _padded_to_packed(x, idx):
    """
    Convert features from padded to packed.

    Args:
      x: (N, V, D)
      idx: LongTensor of shape (VV,)

    Returns:
      feats_packed: (VV, D)
    """

    D = x.shape[-1]
    idx = idx.view(-1, 1).expand(-1, D)
    x_packed = x.view(-1, D).gather(0, idx)
    return x_packed