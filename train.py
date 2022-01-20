# THe training script
import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter

from config import get_sensat_cfg
from dataset.build_data_loader import build_data_loader
from utils.model_record_name import generate_model_record_name


cfg_file = "Sensat_basic.yaml"



if __name__ == "__main__":

    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)
    save_path = generate_model_record_name(cfg)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    shutil.copyfile(cfg_file, os.path.join(save_path,cfg_file))

    worker_id = cfg.SOLVER.GPU_ID
    device = torch.device("cuda:%d" % worker_id)

    loaders = {}
    loaders["train"] = build_data_loader(cfg, "Terrain", "train", num_workers=cfg.DATASETS.NUM_THREADS)