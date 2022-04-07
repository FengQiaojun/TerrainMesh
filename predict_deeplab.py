# Load a PyTorch deeplab and train

import numpy as np
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import get_sensat_cfg
from dataset.build_data_loader import build_data_loader
from loss import FocalLoss
from model.deeplab import deeplabv3_resnet18, deeplabv3_resnet34, deeplabv3_resnet50
from utils.model_record_name import generate_segmodel_record_name
from utils.semantic_labels import convert_class_to_rgb_sensat_simplified
from utils.stream_metrics import StreamSegMetrics

cfg_file = "Sensat_basic.yaml"

if __name__ == "__main__":
    # Load the config and create a folder to save the outputs.
    cfg = get_sensat_cfg()
    cfg.merge_from_file(cfg_file)

    
    # Specify the GPU. Here use 2 of them.
    worker_id = cfg.SOLVER.GPU_ID
    device = torch.device("cuda:%d" % worker_id if torch.cuda.is_available() else 'cpu')
    
    # Build the model
    '''
    if cfg.MODEL.BACKBONE == "resnet50":
        model = deeplabv3_resnet50(cfg)
    elif cfg.MODEL.BACKBONE == "resnet34":
        model = deeplabv3_resnet34(cfg)
    elif cfg.MODEL.BACKBONE == "resnet18":
        model = deeplabv3_resnet18(cfg)
    #model = nn.DataParallel(model)
    '''
    model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
    model.classifier[4] = nn.Conv2d(256, cfg.MODEL.DEEPLAB.NUM_CLASSES, kernel_size=1, stride=1)
    model.to(device)
    
    
    if cfg.MODEL.RESUME:
        save_model_path = cfg.MODEL.RESUME_MODEL
        save_path = os.path.join(cfg.MODEL.RESUME_MODEL,"..")
        #cfg.merge_from_file(os.path.join(save_path,"Sensat_basic.yaml"))
        checkpoint = torch.load(save_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()


    # Set up criterion
    if cfg.MODEL.DEEPLAB.LOSS == "cross_entropy":
        if cfg.MODEL.DEEPLAB.CLASS_WEIGHTED:
            loss_fn = nn.CrossEntropyLoss(weight = torch.Tensor(cfg.MODEL.DEEPLAB.CLASS_WEIGHT).to(device), ignore_index=0, reduction='mean')
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    elif cfg.MODEL.DEEPLAB.LOSS == "focal_loss":
        if cfg.MODEL.DEEPLAB.CLASS_WEIGHTED:
            loss_fn = FocalLoss(weight = torch.Tensor(cfg.MODEL.DEEPLAB.CLASS_WEIGHT).to(device), ignore_index=0, size_average=True)
        else:
            loss_fn = FocalLoss(ignore_index=0, size_average=True)
    metrics = StreamSegMetrics(cfg.MODEL.DEEPLAB.NUM_CLASSES)

    # Build the DataLoaders 
    loaders = {}
    loaders["train"] = build_data_loader(cfg, "SensatSemantic", split_name=cfg.DATASETS.TRAINSET, num_workers=cfg.DATASETS.NUM_THREADS)
    loaders["val"] = build_data_loader(cfg, "SensatSemantic", split_name=cfg.DATASETS.VALSET, num_workers=cfg.DATASETS.NUM_THREADS)
    loaders["test"] = build_data_loader(cfg, "SensatSemantic", split_name=cfg.DATASETS.TESTSET, num_workers=cfg.DATASETS.NUM_THREADS)
    batch_num_train = int(np.ceil(len(loaders["train"].dataset)/loaders["train"].batch_size))
    batch_num_val = int(np.ceil(len(loaders["val"].dataset)/loaders["val"].batch_size))
    batch_num_test = int(np.ceil(len(loaders["test"].dataset)/loaders["test"].batch_size))
    print("Training set size %d. Training batch number %d."%(len(loaders["train"].dataset),batch_num_train))
    print("Validation set size %d. Validation batch number %d."%(len(loaders["val"].dataset),batch_num_val))
    print("Test set size %d. Test batch number %d."%(len(loaders["test"].dataset),batch_num_test))
    
    loss_sum = 0
    num_count = 0
    metrics.reset()

    with torch.no_grad():
        #for i, batch in tqdm(enumerate(loaders["train"]),total=batch_num_train):
        for i, batch in tqdm(enumerate(loaders["val"]),total=batch_num_val):
        #for i, batch in tqdm(enumerate(loaders["test"]),total=batch_num_test):
                    
            rgb_img, sparse_depth, depth_edt, init_mesh_render_depth, gt_semantic = batch 
            # Concatenate the inputs
            if cfg.MODEL.CHANNELS == 3:
                input_img = rgb_img
            elif cfg.MODEL.CHANNELS == 4:
                input_img = torch.cat((rgb_img,init_mesh_render_depth),dim=1)
            elif cfg.MODEL.CHANNELS == 5:
                input_img = torch.cat((rgb_img,init_mesh_render_depth,depth_edt),dim=1)
            input_img = input_img.to(device, dtype=torch.float32)
            gt_semantic = gt_semantic.to(device, dtype=torch.long)
            if cfg.MODEL.BACKBONE == "resnet50":
                pred_semantic = model(input_img)["out"]                
            else:
                pred_semantic = model(input_img)             
            loss = loss_fn(pred_semantic, gt_semantic)
            loss_sum += loss.detach().cpu().numpy()*rgb_img.shape[0]
            num_count += rgb_img.shape[0]
       
            preds = pred_semantic.detach().max(dim=1)[1].cpu().numpy()
            metrics.update(preds, gt_semantic.cpu().numpy())
            
            '''
            for j in range(rgb_img.shape[0]):
                plt.subplot(131)
                plt.imshow(input_img.permute(0,2,3,1).cpu().numpy()[j,::])
                plt.subplot(132)
                plt.imshow(convert_class_to_rgb_sensat_simplified(preds[j,::]))
                plt.subplot(133)
                plt.imshow(convert_class_to_rgb_sensat_simplified(gt_semantic.cpu().numpy()[j,::]))
                plt.show()
            '''
 
        score = metrics.get_results()
        print("Acc",score['Overall Acc'])
        print("Mean Acc",score['Mean Acc'])
        print("MeanIoU",score['Mean IoU'])
        print("Class IoU",score['Class IoU'])
        