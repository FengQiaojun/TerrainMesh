import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SensatSemanticDataset(Dataset):
    def __init__(self,data_dir,split=None,normalize_images=True,):
        transform = [transforms.ToTensor()]
        # do imagenet normalization
        if normalize_images:
            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]
            transform.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.transform = transforms.Compose(transform)
        self.rgb_img_ids = []
        self.sem_img_ids = []
        # split is the name of file containing list of sequence
        if not os.path.isfile(os.path.join(data_dir,split+".txt")):
            print("split %s does not exist! Check again!\n"%split)
            return 0
        with open(os.path.join(data_dir,split+".txt")) as f:
            seq_idx_list = f.read().splitlines()
        for seq in seq_idx_list:
            for target in sorted(os.listdir(os.path.join(data_dir,seq,"Images"))):
                self.rgb_img_ids.append(os.path.join(data_dir,seq,"Images",target))
                self.sem_img_ids.append(os.path.join(data_dir,seq,"Semantics",target))

    def __len__(self):
        return len(self.rgb_img_ids)

    def __getitem__(self, idx):
        rgb_img_path = self.rgb_img_ids[idx]
        rgb_img = Image.open(rgb_img_path).convert("RGB")
        rgb_img = np.float32(np.array(rgb_img)) / 255.
        rgb_img = self.transform(rgb_img)
        sem_img_path = self.sem_img_ids[idx]
        sem_img = Image.open(sem_img_path)
        sem_img = torch.from_numpy(np.array(sem_img))
        return rgb_img, sem_img