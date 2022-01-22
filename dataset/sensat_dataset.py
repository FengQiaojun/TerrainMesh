import logging
import os
import numpy as np
from imageio import imread
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

logger = logging.getLogger(__name__)

# - data_dir the source folder
# - split: ["train","train_birmingham","train_cambridge","val","val_birmingham","val_cambridge","test","test_birmingham","test_cambridge"]
# - meshing: ["mesh576","mesh1024"]
# - samples: [500,1000,2000,4000]
# - depth_scale:
class SensatSemanticDataset(Dataset):
    def __init__(self, data_dir, split=None, meshing=None, samples=None, depth_scale=None, normalized_depth = False, normalize_images=True,):
        transform = [transforms.ToTensor()]
        # do imagenet normalization
        if normalize_images:
            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]
            transform.append(transforms.Normalize(
                mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.transform = transforms.Compose(transform)

        self.split = split
        self.meshing = meshing
        self.samples = samples
        self.depth_scale = depth_scale
        self.normalized_depth = normalized_depth

        self.rgb_img_ids = []
        self.sparse_depth_ids = []
        self.depth_edt_ids = []
        self.init_mesh_ids = []
        self.init_mesh_render_depth_ids = []
        self.gt_depth_ids = []
        self.gt_mesh_pcd_ids = []
        self.sem_img_ids = []

        # split is the name of file containing list of sequence
        if not os.path.isfile(os.path.join(data_dir, "split", split+".txt")):
            print("split %s does not exist! Check again!\n" % split)
            return 0
        with open(os.path.join(data_dir, "split", split+".txt")) as f:
            seq_idx_list = f.read().splitlines()
        for seq in seq_idx_list:
            for target in sorted(os.listdir(os.path.join(data_dir, seq, "Images"))):
                target_idx = target[:-4]
                self.rgb_img_ids.append(os.path.join(
                    data_dir, seq, "Images", target))
                self.sparse_depth_ids.append(os.path.join(
                    data_dir, seq, "Pcds_"+str(samples), target))
                self.depth_edt_ids.append(os.path.join(
                    data_dir, seq, "Pcds_"+str(samples), target_idx+"_edt.pt"))
                self.init_mesh_ids.append(os.path.join(
                    data_dir, seq, "Pcds_"+str(samples), target_idx+"_"+meshing+".obj"))
                self.init_mesh_render_depth_ids.append(os.path.join(
                    data_dir, seq, "Pcds_"+str(samples), target_idx+"_"+meshing+".png"))
                self.gt_depth_ids.append(os.path.join(
                    data_dir, seq, "Depths", target))
                self.gt_mesh_pcd_ids.append(os.path.join(
                    data_dir, seq, "Meshes", target_idx+"_pcd.pt"))
                self.sem_img_ids.append(os.path.join(
                    data_dir, seq, "Semantics_5", target))
# Temporary testing            
#                if len(self.rgb_img_ids) == 16:
#                    break
#            break    
# Temporary testing 

    def __len__(self):
        return len(self.rgb_img_ids)

    def __getitem__(self, idx):
        rgb_img_path = self.rgb_img_ids[idx]
        sparse_depth_path = self.sparse_depth_ids[idx]
        depth_edt_path = self.depth_edt_ids[idx]
        init_mesh_path = self.init_mesh_ids[idx]
        init_mesh_render_depth_path = self.init_mesh_render_depth_ids[idx]
        gt_depth_path = self.gt_depth_ids[idx]
        gt_mesh_pcd_path = self.gt_mesh_pcd_ids[idx]
        sem_img_path = self.sem_img_ids[idx]
        
        rgb_img = np.asfarray(imread(rgb_img_path)/255, dtype=np.float32)
        rgb_img = self.transform(rgb_img)
        # TODO: why divide by 1000?
        depth_input_scale = 1000
        sparse_depth = np.asfarray(
            imread(sparse_depth_path)/self.depth_scale, dtype=np.float32)/depth_input_scale
        if self.normalized_depth:
            depth_available_map = sparse_depth>0
            num_depth = np.sum(depth_available_map)
            mean_depth = np.sum(sparse_depth)/num_depth*depth_input_scale
        sparse_depth = transforms.ToTensor()(sparse_depth)
        # TODO: why divide by 20?
        edt_input_scale = 20
        depth_edt = torch.clamp(torch.load(depth_edt_path).float()/edt_input_scale, min=0, max=2)
        depth_edt = torch.unsqueeze(depth_edt, dim=0)
        init_mesh_v, init_mesh_f, _ = load_obj(
            init_mesh_path, load_textures=False)
        if self.normalized_depth:
            init_mesh_v /= mean_depth
        init_mesh_f = init_mesh_f.verts_idx
        init_mesh_render_depth = np.asfarray(imread(
            init_mesh_render_depth_path)/self.depth_scale, dtype=np.float32)/depth_input_scale
        init_mesh_render_depth = transforms.ToTensor()(init_mesh_render_depth)
        gt_depth = np.asfarray(imread(gt_depth_path) /
                               self.depth_scale, dtype=np.float32)
        if self.normalized_depth:
            gt_depth /= mean_depth
        gt_depth = transforms.ToTensor()(gt_depth)
        gt_mesh_pcd = torch.load(gt_mesh_pcd_path)
        if self.normalized_depth:
            gt_mesh_pcd /= mean_depth
        sem_img = np.asfarray(imread(sem_img_path), dtype=np.int8)
        sem_img = torch.tensor(sem_img, dtype = torch.long)
        return rgb_img, sparse_depth, depth_edt, init_mesh_v, init_mesh_f, init_mesh_render_depth, gt_depth, gt_mesh_pcd, sem_img
        
    # TODO
    @staticmethod
    def collate_fn(batch):
        rgb_img, sparse_depth, depth_edt, init_mesh_v, init_mesh_f, init_mesh_render_depth, gt_depth, gt_mesh_pcd, sem_img = zip(
            *batch)
        rgb_img = torch.stack(rgb_img, dim=0)
        sparse_depth = torch.stack(sparse_depth, dim=0)
        depth_edt = torch.stack(depth_edt, dim=0)
        if init_mesh_v[0] is not None and init_mesh_f[0] is not None:
            init_mesh = Meshes(verts=list(init_mesh_v),
                               faces=list(init_mesh_f),)
        else:
            init_mesh = None
        init_mesh_render_depth = torch.stack(init_mesh_render_depth, dim=0)
        gt_depth = torch.stack(gt_depth, dim=0)
        gt_mesh_pcd = torch.stack(gt_mesh_pcd, dim=0)
        sem_img = torch.stack(sem_img, dim=0)
        return rgb_img, sparse_depth, depth_edt, init_mesh, init_mesh_render_depth, gt_depth, gt_mesh_pcd, sem_img

    def postprocess(self, batch, device=None):
        if device is None:
            device = torch.device("cuda")
        rgb_img, sparse_depth, depth_edt, init_mesh, init_mesh_render_depth, gt_depth, gt_mesh_pcd, sem_img = batch
        rgb_img = rgb_img.to(device)
        sparse_depth = sparse_depth.to(device)
        depth_edt = depth_edt.to(device)
        if init_mesh is not None:
            init_mesh = init_mesh.to(device)
        init_mesh_render_depth = init_mesh_render_depth.to(device)
        gt_depth = gt_depth.to(device)
        if gt_mesh_pcd is not None:
            gt_mesh_pcd = gt_mesh_pcd.to(device)
        sem_img = sem_img.to(device)
        return rgb_img, sparse_depth, depth_edt, init_mesh, init_mesh_render_depth, gt_depth, gt_mesh_pcd, sem_img


# define a data loading function that only read one instance in TerrainDataset(WHU)
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
