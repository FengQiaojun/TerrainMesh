# A simple function that generate folder name based on time and model parameters

import datetime
import os


def generate_model_record_name(cfg, prefix):
    now = datetime.datetime.now()
    supervision = ""
    if cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT > 0:
        supervision+="2D_"
    if cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT > 0:
        supervision+="3D_"
    if cfg.MODEL.SEMANTIC and cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT > 0:
        supervision+="Semantic_"
        supervision+=cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_FUNC+"_"
    supervision = supervision[:-1]
    depth_normalized = "_"
    if cfg.DATASETS.NORMALIZE_MESH:
        depth_normalized = "_dnorm_"
    model_name = "%02d%02d_%02d%02d_" % (now.month, now.day, now.hour, now.minute)+cfg.MODEL.BACKBONE+"_"+cfg.DATASETS.TRAINSET+"_"+cfg.DATASETS.MESHING+"_"+"depth"+str(cfg.DATASETS.SAMPLES)+depth_normalized+supervision+"_"+"channel"+str(cfg.MODEL.CHANNELS)+"_"+str(cfg.MODEL.MESH_HEAD.NUM_STAGES)+"_"+str(cfg.MODEL.MESH_HEAD.NUM_GRAPH_CONVS)+"_"+str(cfg.MODEL.MESH_HEAD.GRAPH_CONV_DIM)+"_"+str(cfg.SOLVER.BASE_LR)
    return os.path.join(prefix, model_name)

def generate_segmodel_record_name(cfg, prefix):
    now = datetime.datetime.now()
    if cfg.DATASETS.NORMALIZE_MESH:
        depth_normalized = "_dnorm_"
    model_name = "%02d%02d_%02d%02d_" % (now.month, now.day, now.hour, now.minute)+"deeplab_"+cfg.MODEL.BACKBONE+"_"+cfg.DATASETS.TRAINSET+"_"+cfg.DATASETS.MESHING+"_"+"depth"+str(cfg.DATASETS.SAMPLES)+"_"+"channel"+str(cfg.MODEL.CHANNELS)+"_"+cfg.MODEL.DEEPLAB.LOSS+"_"+str(cfg.MODEL.DEEPLAB.NUM_EPOCHS)+"_"+str(cfg.MODEL.DEEPLAB.LR)
    return os.path.join(prefix, model_name)

