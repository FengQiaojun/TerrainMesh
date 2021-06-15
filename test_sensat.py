import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import time

from dataset import SensatSemanticDataset
from segmentation import network, utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = SensatSemanticDataset(data_dir="/mnt/NVMe-2TB/qiaojun/SensatUrban",split="train_birmingham",normalize_images=True)
train_loader = data.DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = network.deeplabv3_resnet50(num_classes=14, output_stride=8, pretrained_backbone=False)
model = network.deeplabv3_resnet50(num_classes=14, output_stride=8, pretrained_backbone=True)
utils.set_bn_momentum(model.backbone, momentum=0.01)

gloabl_lr = 0.05

# Set up optimizer
optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*gloabl_lr},
        {'params': model.classifier.parameters(), 'lr': gloabl_lr},
    ], lr=gloabl_lr, momentum=0.9, weight_decay=1e-4)
scheduler = utils.PolyLR(optimizer, 1e5, power=0.9)

# Set up criterion
criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

# Restore
best_score = 0.0
cur_itrs = 0
cur_epochs = 0
print("[!] Retrain")
model = nn.DataParallel(model)
model.to(device)

interval_loss = 0
model.train()
while True: #cur_itrs < opts.total_itrs:
    # =====  Train  =====
    cur_epochs += 1
    for (images, labels) in train_loader:
        
        cur_itrs += 1

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        np_loss = loss.detach().cpu().numpy()
        interval_loss = np_loss
 
        print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, 1e5, interval_loss))
        scheduler.step()  

        img_rgb = images.detach().cpu().numpy()[0,:,:,:].transpose((1,2,0))
        img_label = labels.detach().cpu().numpy()[0,:,:]
        img_predict = outputs.detach().cpu().numpy()[0,:,:]
        img_predict = np.argmax(img_predict,axis=0)
        
        if np_loss < 0.2:
            plt.subplot(131)
            plt.imshow(img_rgb)
            plt.subplot(132)
            plt.imshow(img_label)
            plt.subplot(133)
            plt.imshow(img_predict)
            plt.show()
        
        break