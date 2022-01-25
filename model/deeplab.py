# Some Deeplab models

import torch
import torch.nn as nn
#from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, ASPP

class DeepLabHeadNew(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36], out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, num_classes, 1),
        )

def deeplabv3_resnet50(cfg):
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    if cfg.MODEL.CHANNELS != 3:
        model.backbone.conv1 = nn.Conv2d(cfg.MODEL.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier[4] = nn.Conv2d(256, cfg.MODEL.DEEPLAB.NUM_CLASSES, kernel_size=1, stride=1)
    return model

def deeplabv3_resnet34(cfg):
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    model.backbone = create_feature_extractor(resnet34(pretrained=True),{"layer4": "out"})
    if cfg.MODEL.CHANNELS != 3:
        model.backbone.conv1 = nn.Conv2d(cfg.MODEL.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = DeepLabHeadNew(in_channels=512, out_channels=128, num_classes=cfg.MODEL.DEEPLAB.NUM_CLASSES)
    return model

def deeplabv3_resnet18(cfg):
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    model.backbone = create_feature_extractor(resnet18(pretrained=True),{"layer4": "out"})
    if cfg.MODEL.CHANNELS != 3:
        model.backbone.conv1 = nn.Conv2d(cfg.MODEL.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = DeepLabHeadNew(in_channels=512, out_channels=128, num_classes=cfg.MODEL.DEEPLAB.NUM_CLASSES)
    return model
