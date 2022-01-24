# Load a PyTorch deeplab and train

import torch
import torch.nn as nn

from config import get_sensat_cfg

cfg_file = "Sensat_basic.yaml"

if __name__ == "__main__":
    # Load the config and create a folder to save the outputs.
    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)

    model.backbone.conv1 = nn.Conv2d(cfg.MODEL.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)