# Some Deeplab models

import torch
import torch.nn as nn

def deeplabv3_resnet50(cfg):
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    if cfg.MODEL.CHANNELS != 3:
        model.backbone.conv1 = nn.Conv2d(cfg.MODEL.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier[4] = nn.Conv2d(256, cfg.MODEL.DEEPLAB.NUM_CLASSES, kernel_size=1, stride=1)
    return model