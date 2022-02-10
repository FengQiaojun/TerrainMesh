import logging
import os
import numpy as np
from imageio import imread
from tqdm import tqdm
import torch
import torch.nn as nn
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
class SensatDataset(Dataset):
    def __init__(self, data_dir, split=None, meshing=None, samples=None, depth_scale=None, normalize_mesh = False, normalize_images=True, size=None):
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
        self.normalize_mesh = normalize_mesh

        self.rgb_img_ids = []
        self.sparse_depth_ids = []
        self.depth_edt_ids = []
        self.sem_pred_ids = []
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
                for sam in samples:
                    target_idx = target[:-4]
                    self.rgb_img_ids.append(os.path.join(
                            data_dir, seq, "Images", target))
                    self.sparse_depth_ids.append(os.path.join(
                            data_dir, seq, "Pcds_"+str(sam), target))
                    self.depth_edt_ids.append(os.path.join(
                            data_dir, seq, "Pcds_"+str(sam), target_idx+"_edt.pt"))
                    self.sem_pred_ids.append(os.path.join(
                            data_dir, seq, "Semantics_2D", target_idx+".pt"))                        
                    self.init_mesh_ids.append(os.path.join(
                            data_dir, seq, "Pcds_"+str(sam), target_idx+"_"+meshing+".obj"))
                    self.init_mesh_render_depth_ids.append(os.path.join(
                            data_dir, seq, "Pcds_"+str(sam), target_idx+"_"+meshing+".png"))
                    self.gt_depth_ids.append(os.path.join(
                            data_dir, seq, "Depths", target))
                    self.gt_mesh_pcd_ids.append(os.path.join(
                            data_dir, seq, "Meshes", target_idx+"_pcd.pt"))
                    self.sem_img_ids.append(os.path.join(
                            data_dir, seq, "Semantics_5", target))                        

                    if size is not None and len(self.rgb_img_ids) == size:
                        return 
# Temporary testing            
#                if len(self.rgb_img_ids) == 660:
#                    break
#            break    
# Temporary testing 

    def __len__(self):
        return len(self.rgb_img_ids)

    def __getitem__(self, idx):
        rgb_img_path = self.rgb_img_ids[idx]
        sparse_depth_path = self.sparse_depth_ids[idx]
        depth_edt_path = self.depth_edt_ids[idx]
        sem_pred_path = self.sem_pred_ids[idx]
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
        sparse_depth = transforms.ToTensor()(sparse_depth)
        # TODO: why divide by 20?
        edt_input_scale = 20
        depth_edt = torch.clamp(torch.load(depth_edt_path).float()/edt_input_scale, min=0, max=2)
        depth_edt = torch.unsqueeze(depth_edt, dim=0)
        sem_pred = torch.load(sem_pred_path)
        # TODO: whether to do so?
        #sem_pred = nn.Softmax(dim=0)(sem_pred/100)
        init_mesh_v, init_mesh_f, _ = load_obj(
            init_mesh_path, load_textures=False)
        if self.normalize_mesh:
            init_mesh_scale = torch.mean(init_mesh_v[:,2])
            init_mesh_v /= init_mesh_scale
        else:
            init_mesh_scale = torch.Tensor(1)
        init_mesh_f = init_mesh_f.verts_idx
        init_mesh_render_depth = np.asfarray(imread(
            init_mesh_render_depth_path)/self.depth_scale, dtype=np.float32)/depth_input_scale
        init_mesh_render_depth = transforms.ToTensor()(init_mesh_render_depth)
        gt_depth = np.asfarray(imread(gt_depth_path) /
                               self.depth_scale, dtype=np.float32)
        gt_depth = transforms.ToTensor()(gt_depth)
        gt_mesh_pcd = torch.load(gt_mesh_pcd_path)
        sem_img = np.asfarray(imread(sem_img_path), dtype=np.int8)
        sem_img = torch.tensor(sem_img, dtype = torch.long)
        return rgb_img, sparse_depth, depth_edt, sem_pred, init_mesh_v, init_mesh_f, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, sem_img
        
    # TODO
    @staticmethod
    def collate_fn(batch):
        rgb_img, sparse_depth, depth_edt, sem_pred, init_mesh_v, init_mesh_f, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, sem_img = zip(
            *batch)
        rgb_img = torch.stack(rgb_img, dim=0)
        sparse_depth = torch.stack(sparse_depth, dim=0)
        depth_edt = torch.stack(depth_edt, dim=0)
        sem_pred = torch.stack(sem_pred, dim=0)
        if init_mesh_v[0] is not None and init_mesh_f[0] is not None:
            init_mesh = Meshes(verts=list(init_mesh_v),
                               faces=list(init_mesh_f),)
        else:
            init_mesh = None
        init_mesh_scale = torch.stack(init_mesh_scale, dim=0)
        init_mesh_render_depth = torch.stack(init_mesh_render_depth, dim=0)
        gt_depth = torch.stack(gt_depth, dim=0)
        gt_mesh_pcd = torch.stack(gt_mesh_pcd, dim=0)
        sem_img = torch.stack(sem_img, dim=0)
        return rgb_img, sparse_depth, depth_edt, sem_pred, init_mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, sem_img

    def postprocess(self, batch, device=None):
        if device is None:
            device = torch.device("cuda")
        rgb_img, sparse_depth, depth_edt, sem_pred, init_mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, sem_img = batch
        rgb_img = rgb_img.to(device)
        sparse_depth = sparse_depth.to(device)
        depth_edt = depth_edt.to(device)
        sem_pred = sem_pred.to(device)
        if init_mesh is not None:
            init_mesh = init_mesh.to(device)
        init_mesh_scale = init_mesh_scale.to(device)
        init_mesh_render_depth = init_mesh_render_depth.to(device)
        gt_depth = gt_depth.to(device)
        if gt_mesh_pcd is not None:
            gt_mesh_pcd = gt_mesh_pcd.to(device)
        sem_img = sem_img.to(device)
        return rgb_img, sparse_depth, depth_edt, sem_pred, init_mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, sem_img


class SensatSemanticDataset(Dataset):
    def __init__(self, data_dir, split=None, meshing=None, samples=None, depth_scale=None, normalize_images=True,):
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
        
        self.rgb_img_ids = []
        self.sparse_depth_ids = []
        self.depth_edt_ids = []
        self.init_mesh_render_depth_ids = []
        self.sem_img_ids = []

        # split is the name of file containing list of sequence
        if not os.path.isfile(os.path.join(data_dir, "split", split+".txt")):
            print("split %s does not exist! Check again!\n" % split)
            return 0
        with open(os.path.join(data_dir, "split", split+".txt")) as f:
            seq_idx_list = f.read().splitlines()
        for seq in seq_idx_list:
            for target in sorted(os.listdir(os.path.join(data_dir, seq, "Images"))):
                for sam in samples:
                    target_idx = target[:-4]
                    self.rgb_img_ids.append(os.path.join(
                        data_dir, seq, "Images", target))
                    self.sparse_depth_ids.append(os.path.join(
                        data_dir, seq, "Pcds_"+str(sam), target))
                    self.depth_edt_ids.append(os.path.join(
                        data_dir, seq, "Pcds_"+str(sam), target_idx+"_edt.pt"))
                    self.init_mesh_render_depth_ids.append(os.path.join(
                        data_dir, seq, "Pcds_"+str(sam), target_idx+"_"+meshing+".png"))
                    self.sem_img_ids.append(os.path.join(
                        data_dir, seq, "Semantics_5", target))
# Temporary testing            
#                if len(self.rgb_img_ids) == 660:
#                    break
#            break    
# Temporary testing 

    def __len__(self):
        return len(self.rgb_img_ids)

    def __getitem__(self, idx):
        rgb_img_path = self.rgb_img_ids[idx]
        sparse_depth_path = self.sparse_depth_ids[idx]
        depth_edt_path = self.depth_edt_ids[idx]
        init_mesh_render_depth_path = self.init_mesh_render_depth_ids[idx]
        sem_img_path = self.sem_img_ids[idx]
        
        rgb_img = np.asfarray(imread(rgb_img_path)/255, dtype=np.float32)
        rgb_img = self.transform(rgb_img)
        # TODO: why divide by 1000?
        depth_input_scale = 1000
        sparse_depth = np.asfarray(
            imread(sparse_depth_path)/self.depth_scale, dtype=np.float32)/depth_input_scale
        sparse_depth = transforms.ToTensor()(sparse_depth)
        # TODO: why divide by 20?
        edt_input_scale = 20
        depth_edt = torch.clamp(torch.load(depth_edt_path).float()/edt_input_scale, min=0, max=2)
        depth_edt = torch.unsqueeze(depth_edt, dim=0)
        init_mesh_render_depth = np.asfarray(imread(
            init_mesh_render_depth_path)/self.depth_scale, dtype=np.float32)/depth_input_scale
        init_mesh_render_depth = transforms.ToTensor()(init_mesh_render_depth)
        sem_img = np.asfarray(imread(sem_img_path), dtype=np.int8)
        sem_img = torch.tensor(sem_img, dtype = torch.long)
        return rgb_img, sparse_depth, depth_edt, init_mesh_render_depth, sem_img


# define a data loading function that only read one instance in TerrainDataset(WHU)
def load_data_by_index(cfg,
                       seq_idx="",
                       img_idx="",
                       meshing="",
                       samples="",
                       device=None
                       ):

    data_dir = cfg.DATASETS.DATA_DIR

    root = os.path.join(data_dir, seq_idx)
    rgb_img_path = os.path.join(root, "Images", img_idx+".png")
    sparse_depth_path = os.path.join(root, "Pcds_"+samples, img_idx+".png")
    depth_edt_path = os.path.join(root, "Pcds_"+samples, img_idx+"_edt.pt")
    sem_pred_path = os.path.join(root, "Semantics_2D", img_idx+".pt")
    init_mesh_path = os.path.join(root, "Pcds_"+samples, img_idx+"_"+meshing+".obj")
    init_mesh_render_depth_path = os.path.join(root, "Pcds_"+samples, img_idx+"_"+meshing+".png")
    gt_depth_path = os.path.join(root, "Depths", img_idx+".png")
    gt_mesh_pcd_path = os.path.join(root, "Meshes", img_idx+"_pcd.pt")
    sem_img_path = os.path.join(root, "Semantics_5", img_idx+".png")

    rgb_img = np.asfarray(imread(rgb_img_path)/255, dtype=np.float32)
    rgb_img = transforms.ToTensor()(rgb_img)
    # TODO: why divide by 1000?
    depth_input_scale = 1000
    sparse_depth = np.asfarray(
            imread(sparse_depth_path)/1, dtype=np.float32)/depth_input_scale
    sparse_depth = transforms.ToTensor()(sparse_depth)
    # TODO: why divide by 20?
    edt_input_scale = 20
    depth_edt = torch.clamp(torch.load(depth_edt_path).float()/edt_input_scale, min=0, max=2)
    depth_edt = torch.unsqueeze(depth_edt, dim=0)
    sem_pred = torch.load(sem_pred_path)
    # TODO: whether to do so?
    sem_pred = nn.Softmax(dim=0)(sem_pred/1)
    init_mesh_v, init_mesh_f, _ = load_obj(
        init_mesh_path, load_textures=False)
    
    init_mesh_scale = torch.mean(init_mesh_v[:,2])
    init_mesh_v /= init_mesh_scale

    init_mesh_f = init_mesh_f.verts_idx
    init_mesh_render_depth = np.asfarray(imread(
            init_mesh_render_depth_path) / 100, dtype=np.float32)/depth_input_scale
    init_mesh_render_depth = transforms.ToTensor()(init_mesh_render_depth)
    gt_depth = np.asfarray(imread(gt_depth_path) / 100, dtype=np.float32)
    gt_depth = transforms.ToTensor()(gt_depth)
    gt_mesh_pcd = torch.load(gt_mesh_pcd_path)
    sem_img = np.asfarray(imread(sem_img_path), dtype=np.int8)
    sem_img = torch.tensor(sem_img, dtype = torch.long)

    rgb_img = torch.unsqueeze(rgb_img, dim=0)
    sparse_depth = torch.unsqueeze(sparse_depth, dim=0)
    depth_edt = torch.unsqueeze(depth_edt, dim=0)
    sem_pred = torch.unsqueeze(sem_pred, dim=0)
    init_mesh = Meshes(verts=[init_mesh_v], faces=[init_mesh_f])
    init_mesh_scale = torch.unsqueeze(init_mesh_scale, dim=0)
    init_mesh_render_depth = torch.unsqueeze(init_mesh_render_depth, dim=0)
    gt_depth = torch.unsqueeze(gt_depth, dim=0)
    gt_mesh_pcd = torch.unsqueeze(gt_mesh_pcd, dim=0)
    sem_img = torch.unsqueeze(sem_img, dim=0)


    rgb_img = rgb_img.to(device)
    sparse_depth = sparse_depth.to(device)
    depth_edt = depth_edt.to(device)
    sem_pred = sem_pred.to(device)
    if init_mesh is not None:
        init_mesh = init_mesh.to(device)
    init_mesh_scale = init_mesh_scale.to(device)
    init_mesh_render_depth = init_mesh_render_depth.to(device)
    gt_depth = gt_depth.to(device)
    if gt_mesh_pcd is not None:
        gt_mesh_pcd = gt_mesh_pcd.to(device)
    sem_img = sem_img.to(device)

    return rgb_img, sparse_depth, depth_edt, sem_pred, init_mesh, init_mesh_scale, init_mesh_render_depth, gt_depth, gt_mesh_pcd, sem_img