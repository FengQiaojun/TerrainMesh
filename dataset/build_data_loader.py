# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from .terrain_dataset import TerrainDataset
from .sensat_dataset import SensatDataset, SensatSemanticDataset

logger = logging.getLogger(__name__)


def _identity(x):
    return x


def build_data_loader(
    cfg, dataset, split_name, num_workers=8, multigpu=False, shuffle=True
):

    batch_size = cfg.SOLVER.BATCH_SIZE
    if split_name in ["train_eval", "val"]:
        batch_size = cfg.SOLVER.BATCH_SIZE_EVAL
    elif split_name == "test":
        batch_size = cfg.SOLVER.BATCH_SIZE_EVAL

    num_gpus = 1
    batch_size //= num_gpus

    logger.info('Building dataset for split "%s"' % split_name)
    if dataset == "Terrain":
        dset = TerrainDataset(
            data_dir=cfg.DATASETS.DATA_DIR,
            split=split_name,
            samples=cfg.DATASETS.SAMPLES,
            normalize_images=cfg.DATASETS.NORMALIZE_IMAGES,
            meshing=cfg.DATASETS.MESHING,
            noise=cfg.DATASETS.NOISE
        )
        collate_fn = TerrainDataset.collate_fn
        loader_kwargs = {"batch_size": batch_size,
                     "collate_fn": collate_fn, "num_workers": num_workers}
    elif dataset == "Sensat":
        dset = SensatDataset(
            data_dir=cfg.DATASETS.DATA_DIR,
            split=split_name,
            meshing=cfg.DATASETS.MESHING,
            samples=cfg.DATASETS.SAMPLES,
            depth_scale=cfg.DATASETS.DEPTH_SCALE,
            normalize_mesh=cfg.DATASETS.NORMALIZE_MESH,
            initialize_mesh=cfg.DATASETS.INITIALIZE_MESH,
            normalize_images=cfg.DATASETS.NORMALIZE_IMAGES,
            size=cfg.DATASETS.SIZE if cfg.DATASETS.SIZE!=0 else None
        )
        collate_fn = SensatDataset.collate_fn
        loader_kwargs = {"batch_size": batch_size,
                     "collate_fn": collate_fn, "num_workers": num_workers}
    elif dataset == "SensatSemantic":
        dset = SensatSemanticDataset(
            data_dir=cfg.DATASETS.DATA_DIR,
            split=split_name,
            meshing=cfg.DATASETS.MESHING,
            samples=cfg.DATASETS.SAMPLES,
            depth_scale=cfg.DATASETS.DEPTH_SCALE,
            normalize_images=cfg.DATASETS.NORMALIZE_IMAGES,
        )
        loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers}
    
    else:
        raise ValueError("Dataset %s not registered" % dataset)

    if hasattr(dset, "postprocess"):
        postprocess_fn = dset.postprocess
    else:
        postprocess_fn = _identity

    # Right now we only do evaluation with a single GPU on the main process,
    # so only use a DistributedSampler for the training set.
    # TODO: Change this once we do evaluation on multiple GPUs
    if multigpu:
        assert shuffle, "Cannot sample sequentially with distributed training"
        sampler = DistributedSampler(dset)
    else:
        if shuffle:
            sampler = RandomSampler(dset)
        else:
            sampler = SequentialSampler(dset)
    loader_kwargs["sampler"] = sampler
    loader = DataLoader(dset, **loader_kwargs)

    # WARNING this is really gross! We want to access the underlying
    # dataset.postprocess method so we can run it on the main Python process,
    # but the dataset might be wrapped in a Subset instance, or may not even
    # define a postprocess method at all. To get around this we monkeypatch
    # the DataLoader object with the postprocess function we want; this will
    # be a bound method of the underlying Dataset, or an identity function.
    # Maybe it would be cleaner to subclass DataLoader for this?
    loader.postprocess = postprocess_fn

    return loader
