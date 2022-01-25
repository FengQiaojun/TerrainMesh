import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm
import time
from pytorch3d.ops import vert_align

from dataset import SensatDataset
from segmentation import network, utils
from utils.semantic_labels import convert_class_to_rgb_sensat_full
from mesh_init.meshing import regular_512_576, regular_512_1024
from mesh_init.mesh_renderer import render_mesh_vertex_texture
from dataset.build_data_loader import build_data_loader
from config import get_sensat_cfg
from utils.project_verts import project_verts

cam_c = 256
cam_f = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#train_dataset = SensatDataset(data_dir="/mnt/NVMe-2TB/qiaojun/SensatUrban",split="train_birmingham",samples = 1000,meshing = "mesh576",depth_scale=100,normalize_images=True)
#train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
cfg = get_sensat_cfg()
cfg_file = "Sensat_basic.yaml"
cfg.merge_from_file(cfg_file)
cfg.SOLVER.BATCH_SIZE = 2
cfg.DATASETS.SHUFFLE = False
train_loader = build_data_loader(cfg, "Sensat", split_name="train_birmingham", shuffle=False)
print("train_loader successful")
batch_num_train = int(np.ceil(len(train_loader.dataset)/train_loader.batch_size))
print("Training set size %d"%len(train_loader.dataset))
print("batch_num_train %d"%batch_num_train)
     

#model = network.deeplabv3_resnet50(num_classes=14, output_stride=8, pretrained_backbone=False)
model = network.deeplabv3_resnet50(num_classes=14, output_stride=8, pretrained_backbone=True)
utils.set_bn_momentum(model.backbone, momentum=0.01)

gloabl_lr = 0.01
total_epochs = 1

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

#model = nn.DataParallel(model)
model.to(device)

interval_loss = 0.
epoch_loss = 0
model.eval()

while cur_epochs < total_epochs:
    # =====  evaluation  =====
    cur_epochs += 1
    cur_itrs = 0
    for i, batch in tqdm(enumerate(train_loader),total=batch_num_train):
        batch = train_loader.postprocess(batch, device)
        print("after batch")
        rgb_img, sparse_depth, depth_edt, init_mesh, init_mesh_render_depth, gt_depth, gt_mesh_pcd, sem_img = batch
        images = rgb_img
        labels = sem_img
        cur_itrs += 1

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(images)
        
        img_predict_2d_class = outputs.detach().cpu().numpy()[0,:,:]
        img_predict_2d = np.argmax(img_predict_2d_class,axis=0)

        # step 1: get the normalized mesh vertices and 
        vertices,faces,_ = regular_512_1024()
        vertices = (vertices-cam_c)/cam_f
        vertices = np.hstack((vertices,np.ones((vertices.shape[0],1))))
        vertices = torch.Tensor(vertices).unsqueeze(0).to(device)
        faces = torch.Tensor(faces).unsqueeze(0).to(device)
        # step 2: get the feature map
        print("outputs.shape",outputs.shape)
        features = outputs[0,::]
        features = features[None, :]
        print("features.shape",features.shape)
        # step 3: retrieve the per-vertex features
        P = [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        P = torch.tensor(P)[None].repeat(1, 1, 1).to(device)
        vert_pos_packed = project_verts(vertices, P)

        vert_align_feats = vert_align(features, vert_pos_packed)
        # step 4: render the mesh for 2D image
        mesh_img, mesh_depth = render_mesh_vertex_texture(verts=vertices[0,:],faces=faces[0,:],feats=vert_align_feats,image_size=512,focal_length=-2,device=device)
        print(mesh_img.shape)
        img_predict = mesh_img
        img_predict = img_predict.detach().cpu().numpy()[0,:,:]
        img_predict = np.argmax(img_predict,axis=-1)
        print(img_predict.shape)
        img_label = labels.detach().cpu().numpy()[0,0,:,:] 
        plt.subplot(131)
        plt.imshow(img_label)
        plt.subplot(132)
        plt.imshow(img_predict_2d)
        plt.subplot(133)
        plt.imshow(img_predict)
        plt.show()
        
        Image.fromarray(convert_class_to_rgb_sensat_full(img_label)).save('%d_gt.png'%cur_itrs)
        Image.fromarray(convert_class_to_rgb_sensat_full(img_predict_2d)).save('%d_2d.png'%cur_itrs)
        Image.fromarray(convert_class_to_rgb_sensat_full(img_predict)).save('%d_mesh1024_label.png'%cur_itrs)

        '''
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
        Image.fromarray(convert_class_to_rgb_sensat_full(img_label)).save('label_gt.png')
        Image.fromarray(convert_class_to_rgb_sensat_full(img_predict)).save('label_predict.png')
        '''
    break