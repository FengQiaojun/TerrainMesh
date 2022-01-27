# Some Deeplab models

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, ASPP
from .image_backbone import ResNetBackbone

class DeepLabHeadNew(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36], out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, num_classes, 1),
        )

class DeepLabV3New(nn.Module):
    def __init__(self, backbone, classifier):
        super(DeepLabV3New, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)[3]
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result = x
        return result    

def deeplabv3_resnet50(cfg, in_channels=3):
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=cfg.MODEL.DEEPLAB.PRETRAIN)
    if cfg.MODEL.CHANNELS != 3:
        model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier[4] = nn.Conv2d(256, cfg.MODEL.DEEPLAB.NUM_CLASSES, kernel_size=1, stride=1)
    return model

def deeplabv3_resnet34(cfg, in_channels=3):
    cnn = getattr(torchvision.models, "resnet34")(pretrained=cfg.MODEL.DEEPLAB.PRETRAIN)
    backbone = ResNetBackbone(cnn, in_channels)
    classifier = DeepLabHeadNew(in_channels=512, out_channels=128, num_classes=cfg.MODEL.DEEPLAB.NUM_CLASSES)   
    model = DeepLabV3New(backbone=backbone,classifier=classifier)
    return model

def deeplabv3_resnet18(cfg, in_channels=3):
    cnn = getattr(torchvision.models, "resnet18")(pretrained=cfg.MODEL.DEEPLAB.PRETRAIN)
    backbone = ResNetBackbone(cnn, in_channels)
    classifier = DeepLabHeadNew(in_channels=512, out_channels=128, num_classes=cfg.MODEL.DEEPLAB.NUM_CLASSES)   
    model = DeepLabV3New(backbone=backbone,classifier=classifier)
    return model
