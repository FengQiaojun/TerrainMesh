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
        return torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

#optim.SGD([
#                {'params': model.base.parameters()},
#                {'params': model.classifier.parameters(), 'lr': 1e-3}
#            ], lr=1e-2, momentum=0.9)