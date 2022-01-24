import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

from dataset import SensatDataset
from segmentation import network, utils
from utils import denormalize,convert_class_to_rgb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = SensatDataset(data_dir="/mnt/NVMe-2TB/qiaojun/SensatUrban",split="train_birmingham",normalize_images=True)
train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = network.deeplabv3_resnet50(num_classes=14, output_stride=8, pretrained_backbone=False)
model = network.deeplabv3_resnet50(num_classes=14, output_stride=8, pretrained_backbone=True)
utils.set_bn_momentum(model.backbone, momentum=0.01)

gloabl_lr = 0.01
total_epochs = 100

# Set up optimizer
optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*gloabl_lr},
        {'params': model.classifier.parameters(), 'lr': gloabl_lr},
    ], lr=gloabl_lr, momentum=0.9, weight_decay=1e-4)
scheduler = utils.PolyLR(optimizer, 1e5, power=0.9)

def save_ckpt(path):
    """ save current model
    """
    torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
    }, path)
    print("Model saved as %s" % path)

# Set up criterion
criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

# Restore
best_score = 0.0
cur_itrs = 0
cur_epochs = 0
print("[!] Retrain")

checkpoint = torch.load("checkpoints/0615/naive_100.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state"])

model = nn.DataParallel(model)
model.to(device)

interval_loss = 0.
epoch_loss = 0
model.eval()

while cur_epochs < total_epochs:
    # =====  evaluation  =====
    cur_epochs += 1
    cur_itrs = 0
    for (images, labels) in train_loader:
        
        cur_itrs += 1

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(images)
        
        img_rgb = denormalize(images.detach().cpu(),mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]).numpy()[0,:,:,:].transpose((1,2,0))
        img_label = labels.detach().cpu().numpy()[0,:,:]
        img_predict = outputs.detach().cpu().numpy()[0,:,:]
        img_predict = np.argmax(img_predict,axis=0)
        
        
        plt.subplot(131)
        plt.imshow(img_rgb)
        plt.subplot(132)
        plt.imshow(img_label)
        plt.subplot(133)
        plt.imshow(img_predict)
        plt.show()
        
        Image.fromarray((img_rgb * 255).astype(np.uint8)).save('img_rgb.png')
        Image.fromarray(convert_class_to_rgb(img_label)).save('label_gt.png')
        Image.fromarray(convert_class_to_rgb(img_predict)).save('label_predict.png')
        