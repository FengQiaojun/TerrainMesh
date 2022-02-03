import torch.nn as nn
import torchvision

class ResNetBackbone(nn.Module):
    def __init__(self, net, in_channels=3):
        super(ResNetBackbone, self).__init__()
        if in_channels == 3:
            self.conv1 = net.conv1
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
        self.stem = nn.Sequential(self.conv1, net.bn1, net.relu, net.maxpool)
        self.stage1 = net.layer1
        self.stage2 = net.layer2
        self.stage3 = net.layer3
        self.stage4 = net.layer4

    def forward(self, imgs):
        feats = self.stem(imgs)
        conv1 = self.stage1(feats)  # 18, 34: 64
        conv2 = self.stage2(conv1)
        conv3 = self.stage3(conv2)
        conv4 = self.stage4(conv3)

        return [conv1, conv2, conv3, conv4]

class ResNetBackboneCopy(nn.Module):
    def __init__(self, net, in_channels=3):
        super(ResNetBackboneCopy, self).__init__()
        if in_channels == 3:
            self.stem = net.stem
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
            self.stem = nn.Sequential(self.conv1, nn.BatchNorm2d(num_features=64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage1 = net.stage1
        self.stage2 = net.stage2
        self.stage3 = net.stage3
        self.stage4 = net.stage4

    def forward(self, imgs):
        feats = self.stem(imgs)
        conv1 = self.stage1(feats)  # 18, 34: 64
        conv2 = self.stage2(conv1)
        conv3 = self.stage3(conv2)
        conv4 = self.stage4(conv3)

        return [conv1, conv2, conv3, conv4]

_FEAT_DIMS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
    "resnet152": (256, 512, 1024, 2048),
}


def build_backbone(name, in_channels, ref_model, pretrained=True):
    resnets = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    if name in resnets and name in _FEAT_DIMS:
        if ref_model is not None:
            backbone = ResNetBackboneCopy(ref_model,in_channels)
        else:
            cnn = getattr(torchvision.models, name)(pretrained=pretrained)
            backbone = ResNetBackbone(cnn,in_channels)
        '''
        # disable the running average
        for child in backbone.children():
            if type(child) == nn.Conv2d:
                continue
            for ii in range(len(child)):
                if type(child[ii])==nn.BatchNorm2d:
                    child[ii].track_running_stats = False
        '''
        feat_dims = _FEAT_DIMS[name]
        return backbone, feat_dims
    else:
        raise ValueError('Unrecognized backbone type "%s"' % name)