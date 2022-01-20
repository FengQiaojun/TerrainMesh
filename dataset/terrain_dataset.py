import json
import logging
import os
import numpy as np
from imageio import imread
import time
import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from torch.utils.data import Dataset

import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)


class TerrainDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split=None,
        samples=1024,
        normalize_images=True,
        meshing="regular",
        noise=False
    ):
        super(TerrainDataset, self).__init__()

        transform = [T.ToTensor()]
        if normalize_images:
            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]
            transform.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.transform = T.Compose(transform)

        if meshing == "regular":
            init_mesh_ext = "_mesh_render"
        elif meshing == "regular576":
            init_mesh_ext = "_mesh576_render"
        elif meshing == "superpixel":
            init_mesh_ext = "_spmesh_render"
        elif meshing == "sfm":
            init_mesh_ext = "_sfmmesh_render"
        else:
            raise ValueError(
                "Only accept meshing = [\"regular\",\"superpixel\",\"sfm\"]")

        if samples not in [2048, 1024, 512]:
            raise ValueError("Only accept samples = [512,1024,2048]")
        if split not in ["train", "val", "test"]:
            raise ValueError(
                "Only accept split = [\"train\",\"val\",\"test\"]")

        self.rgb_img_ids = []
        self.sparse_depth_ids = []
        self.mesh_render_depth_ids = []
        self.depth_edt_ids = []
        self.gt_depth_ids = []
        self.init_mesh_ids = []
        self.gt_mesh_ids = []
        self.data_ids = []

        self.root = os.path.join(data_dir, split)
        seq_idx_list = os.listdir(self.root)
        #seq_idx_list = ["003_100"]
        for seq in seq_idx_list:
            for target in sorted(os.listdir(os.path.join(self.root, seq, "Images"))):
                target_idx = target[:-4]
                if noise:
                    self.rgb_img_ids.append(os.path.join(
                        self.root, seq, "Images", target))
                    self.sparse_depth_ids.append(os.path.join(
                        self.root, seq, "Pcds_"+str(samples), target))
                    self.mesh_render_depth_ids.append(os.path.join(
                        self.root, seq, "Pcds_"+str(samples), target_idx+init_mesh_ext+".png"))
                    self.depth_edt_ids.append(os.path.join(
                        self.root, seq, "Pcds_"+str(samples), target_idx+"_edt.pt"))
                    self.gt_depth_ids.append(os.path.join(
                        self.root, seq, "Depths", target))
                    self.init_mesh_ids.append(os.path.join(
                        self.root, seq, "Pcds_"+str(samples), target_idx+init_mesh_ext+".obj"))
                    self.gt_mesh_ids.append(os.path.join(
                        self.root, seq, "Meshes", target_idx+".pt"))
                    self.data_ids.append(os.path.join(seq, target_idx))
                else:
                    self.rgb_img_ids.append(os.path.join(
                        self.root, seq, "Images", target))
                    self.sparse_depth_ids.append(os.path.join(
                        self.root, seq, "Pcds_"+str(samples)+"_gt", target))
                    self.mesh_render_depth_ids.append(os.path.join(
                        self.root, seq, "Pcds_"+str(samples)+"_gt", target_idx+init_mesh_ext+".png"))
                    self.depth_edt_ids.append(os.path.join(
                        self.root, seq, "Pcds_"+str(samples)+"_gt", target_idx+"_edt.pt"))
                    self.gt_depth_ids.append(os.path.join(
                        self.root, seq, "Depths", target))
                    self.init_mesh_ids.append(os.path.join(
                        self.root, seq, "Pcds_"+str(samples)+"_gt", target_idx+init_mesh_ext+".obj"))
                    self.gt_mesh_ids.append(os.path.join(
                        self.root, seq, "Meshes", target_idx+".pt"))
                    self.data_ids.append(os.path.join(seq, target_idx))

    def __len__(self):
        return len(self.rgb_img_ids)

    def __getitem__(self, idx):
        # rgb_img, sparse_depth, mesh_render_depth, init_mesh, gt_depth, gt_mesh
        rgb_img_path = self.rgb_img_ids[idx]
        sparse_depth_path = self.sparse_depth_ids[idx]
        mesh_render_depth_path = self.mesh_render_depth_ids[idx]
        depth_edt_path = self.depth_edt_ids[idx]
        init_mesh = self.init_mesh_ids[idx]
        gt_depth_path = self.gt_depth_ids[idx]
        gt_mesh_path = self.gt_mesh_ids[idx]

        rgb_img = np.asfarray(imread(rgb_img_path)/255, dtype=np.float32)
        rgb_img = self.transform(rgb_img)
        # TODO: why divide by 1000?
        sparse_depth = np.asfarray(
            imread(sparse_depth_path)/64, dtype=np.float32)/1000
        sparse_depth = T.ToTensor()(sparse_depth)
        mesh_render_depth = np.asfarray(
            imread(mesh_render_depth_path)/64, dtype=np.float32)/1000
        mesh_render_depth = T.ToTensor()(mesh_render_depth)
        depth_edt = torch.clamp(torch.load(
            depth_edt_path).float()/20, min=0, max=2)
        depth_edt = torch.unsqueeze(depth_edt, dim=0)

        gt_depth = np.asfarray(imread(gt_depth_path)/64, dtype=np.float32)
        gt_depth = T.ToTensor()(gt_depth)

        init_mesh_v, init_mesh_f, _ = load_obj(init_mesh, load_textures=False)
        init_mesh_f = init_mesh_f.verts_idx
        gt_mesh = torch.load(gt_mesh_path)

        return rgb_img, sparse_depth, mesh_render_depth, depth_edt, gt_depth, init_mesh_v, init_mesh_f, gt_mesh, self.data_ids[idx]

    @staticmethod
    def collate_fn(batch):
        rgb_img, sparse_depth, mesh_render_depth, depth_edt, gt_depth, init_mesh_v, init_mesh_f, gt_mesh, data_id = zip(
            *batch)

        rgb_img = torch.stack(rgb_img, dim=0)
        sparse_depth = torch.stack(sparse_depth, dim=0)
        mesh_render_depth = torch.stack(mesh_render_depth, dim=0)
        depth_edt = torch.stack(depth_edt, dim=0)
        gt_depth = torch.stack(gt_depth, dim=0)

        if init_mesh_v[0] is not None and init_mesh_f[0] is not None:
            init_mesh = Meshes(verts=list(init_mesh_v),
                               faces=list(init_mesh_f))
        else:
            init_mesh = None
        gt_mesh = torch.stack(gt_mesh, dim=0)

        return rgb_img, sparse_depth, mesh_render_depth, depth_edt, gt_depth, init_mesh, gt_mesh, data_id

    def postprocess(self, batch, device=None):
        if device is None:
            device = torch.device("cuda")
        rgb_img, sparse_depth, mesh_render_depth, depth_edt, gt_depth, init_mesh, gt_mesh, data_id = batch
        rgb_img = rgb_img.to(device)
        sparse_depth = sparse_depth.to(device)
        mesh_render_depth = mesh_render_depth.to(device)
        depth_edt = depth_edt.to(device)
        gt_depth = gt_depth.to(device)
        if init_mesh is not None:
            init_mesh = init_mesh.to(device)
        if gt_mesh is not None:
            gt_mesh = gt_mesh.to(device)
        return rgb_img, sparse_depth, mesh_render_depth, depth_edt, gt_depth, init_mesh, gt_mesh, data_id


# define a data loading function that only read one instance
def load_data_by_index(cfg,
                       split="",
                       seq_idx="",
                       img_idx="",
                       device=None
                       ):
    data_dir = cfg.DATASETS.DATA_DIR
    split = split
    samples = cfg.DATASETS.SAMPLES
    meshing = cfg.DATASETS.MESHING
    noise = cfg.DATASETS.NOISE

    root = os.path.join(data_dir, split)
    rgb_img_path = os.path.join(root, seq_idx, "Images", img_idx+".png")
    gt_depth_path = os.path.join(root, seq_idx, "Depths", img_idx+".png")
    gt_mesh_path = os.path.join(root, seq_idx, "Meshes", img_idx+".pt")

    if meshing == "regular":
            init_mesh_ext = "_mesh_render"
    elif meshing == "regular576":
            init_mesh_ext = "_mesh576_render"
    elif meshing == "superpixel":
            init_mesh_ext = "_spmesh_render"
    elif meshing == "sfm":
            init_mesh_ext = "_sfmmesh_render"
    else:
            raise ValueError(
                "Only accept meshing = [\"regular\",\"superpixel\",\"sfm\"]")

    if noise:
            sparse_depth_path = os.path.join(
                root, seq_idx, "Pcds_"+str(samples), img_idx+".png")
            mesh_render_depth_path = os.path.join(
                root, seq_idx, "Pcds_"+str(samples), img_idx+init_mesh_ext+".png")
            depth_edt_path = os.path.join(
                root, seq_idx, "Pcds_"+str(samples), img_idx+"_edt.pt")
            init_mesh = os.path.join(
                root, seq_idx, "Pcds_"+str(samples), img_idx+init_mesh_ext+".obj")
    else:
            sparse_depth_path = os.path.join(
                root, seq_idx, "Pcds_"+str(samples)+"_gt", img_idx+".png")
            mesh_render_depth_path = os.path.join(
                root, seq_idx, "Pcds_"+str(samples)+"_gt", img_idx+init_mesh_ext+".png")
            depth_edt_path = os.path.join(
                root, seq_idx, "Pcds_"+str(samples)+"_gt", img_idx+"_edt.pt")
            init_mesh = os.path.join(
                root, seq_idx, "Pcds_"+str(samples)+"_gt", img_idx+init_mesh_ext+".obj")

    rgb_img = np.asfarray(imread(rgb_img_path)/255, dtype=np.float32)
    rgb_img = T.ToTensor()(rgb_img)
    # TODO: why divide by 1000?
    sparse_depth = np.asfarray(
            imread(sparse_depth_path)/64, dtype=np.float32)/1000
    sparse_depth = T.ToTensor()(sparse_depth)
    mesh_render_depth = np.asfarray(
            imread(mesh_render_depth_path)/64, dtype=np.float32)/1000
    mesh_render_depth = T.ToTensor()(mesh_render_depth)
    depth_edt = torch.clamp(torch.load(
            depth_edt_path).float()/20, min=0, max=2)
    depth_edt = torch.unsqueeze(depth_edt, dim=0)
    gt_depth = np.asfarray(imread(gt_depth_path)/64, dtype=np.float32)
    gt_depth = T.ToTensor()(gt_depth)
    init_mesh_v, init_mesh_f, _ = load_obj(init_mesh, load_textures=False)
    init_mesh_f = init_mesh_f.verts_idx
    init_mesh = Meshes(verts=[init_mesh_v], faces=[init_mesh_f])
    gt_mesh = torch.load(gt_mesh_path)

    rgb_img = torch.unsqueeze(rgb_img, dim=0)
    sparse_depth = torch.unsqueeze(sparse_depth, dim=0)
    mesh_render_depth = torch.unsqueeze(mesh_render_depth, dim=0)
    depth_edt = torch.unsqueeze(depth_edt, dim=0)
    gt_depth = torch.unsqueeze(gt_depth, dim=0)
    gt_mesh = torch.unsqueeze(gt_mesh, dim=0)

    if device is None:
            device = torch.device("cuda")
    rgb_img = rgb_img.to(device)
    sparse_depth = sparse_depth.to(device)
    mesh_render_depth = mesh_render_depth.to(device)
    depth_edt = depth_edt.to(device)
    gt_depth = gt_depth.to(device)
    if init_mesh is not None:
            init_mesh = init_mesh.to(device)
    if gt_mesh is not None:
            gt_mesh = gt_mesh.to(device)
    return rgb_img, sparse_depth, mesh_render_depth, depth_edt, gt_depth, init_mesh, gt_mesh
