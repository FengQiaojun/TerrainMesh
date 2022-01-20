# A simple function that generate folder name based on time and model parameters

import datetime
import os 

def generate_model_record_name(cfg,prefix):
    now = datetime.datetime.now()
    model_name = "%02d%02d_%02d%02d_" % (now.month, now.day, now.hour, now.minute)+str(cfg.DATASETS.SAMPLES)+"_"+cfg.DATASETS.MESHING+"_"+cfg.MODEL.BACKBONE+"_"+str(cfg.MODEL.CHANNELS)+"_" + \
        cfg.MODEL.MESH_HEAD.SUPERVISION+"_"+str(cfg.MODEL.MESH_HEAD.NUM_STAGES)+"_"+str(
            cfg.MODEL.MESH_HEAD.NUM_GRAPH_CONVS)+"_"+str(cfg.MODEL.MESH_HEAD.GRAPH_CONV_DIM)+"_"+str(cfg.SOLVER.BASE_LR)
    return os.path.join(prefix,model_name)
