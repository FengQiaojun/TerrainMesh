# THe training script
import os
import shutil

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from config import get_sensat_cfg
from dataset.build_data_loader import build_data_loader
from model.models import VoxMeshHead
from optimizer import build_optimizer
from utils.model_record_name import generate_model_record_name


cfg_file = "Sensat_basic.yaml"



if __name__ == "__main__":
    # Load the config and create a folder to save the outputs.
    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)
    save_path = generate_model_record_name(cfg,prefix="checkpoints")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    shutil.copyfile(cfg_file, os.path.join(save_path,cfg_file))

    # Specify the GPU
    worker_id = cfg.SOLVER.GPU_ID
    device = torch.device("cuda:%d" % worker_id)

    # Build the DataLoaders 
    loaders = {}
    loaders["train"] = build_data_loader(cfg, "Sensat", "train", num_workers=cfg.DATASETS.NUM_THREADS)
    loaders["val"] = build_data_loader(cfg, "Sensat", "val", num_workers=cfg.DATASETS.NUM_THREADS)

    # Build the model
    model = VoxMeshHead(cfg)
    model.to(device)
    # Build the optimizer
    optimizer = build_optimizer(cfg, model)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, threshold=1e-3)

    