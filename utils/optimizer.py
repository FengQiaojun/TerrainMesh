# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

def build_optimizer(cfg, model):
    # TODO add weight decay?
    name = cfg.SOLVER.OPTIMIZER
    lr = cfg.SOLVER.BASE_LR
    momentum = cfg.SOLVER.MOMENTUM
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)