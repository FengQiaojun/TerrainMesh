import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader

from dataset_sensat import SensatSemanticDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = SensatSemanticDataset(data_dir="/media/qiaojun/ssd/Terrain/SensatUrban",split="train",normalize_images=False)
'''
img_1, img_2 = dataset[0]
print(type(img_1),img_1.shape)
print(type(img_2),img_2.shape)
plt.subplot(121)
plt.imshow(np.array(img_1).transpose((1,2,0)))
plt.subplot(122)
plt.imshow(np.array(img_2))
plt.show()
'''
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)

for (images, labels) in train_loader:
    images = images.to(device, dtype=torch.float32)
    labels = labels.to(device, dtype=torch.long)
    break