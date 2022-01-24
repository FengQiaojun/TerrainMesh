# the file to list the all necessary configurations for training/testing.

from fvcore.common.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
def get_sensat_cfg():
    cfg = CN()

    # ------------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------------ #
    cfg.MODEL = CN()
    cfg.MODEL.BACKBONE = "resnet50"
    cfg.MODEL.VOXEL_ON = False
    cfg.MODEL.MESH_ON = False
    cfg.MODEL.CHANNELS = 3

    # ------------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------------ #
    cfg.MODEL.CHECKPOINT = "./checkpoints"  # path to checkpoint

    # ------------------------------------------------------------------------ #
    # Mesh Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.MESH_HEAD = CN()
    cfg.MODEL.MESH_HEAD.NAME = "VoxMeshHead"
    # Numer of stages
    cfg.MODEL.MESH_HEAD.NUM_STAGES = 1
    cfg.MODEL.MESH_HEAD.NUM_GRAPH_CONVS = 1  # per stage
    cfg.MODEL.MESH_HEAD.GRAPH_CONV_DIM = 256
    cfg.MODEL.MESH_HEAD.GRAPH_CONV_INIT = "normal"
    # Mesh sampling
    cfg.MODEL.MESH_HEAD.GT_NUM_SAMPLES = 5000
    cfg.MODEL.MESH_HEAD.PRED_NUM_SAMPLES = 5000
    # loss weights
    cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT = 1.0
    cfg.MODEL.MESH_HEAD.NORMALS_LOSS_WEIGHT = 1.0
    cfg.MODEL.MESH_HEAD.EDGE_LOSS_WEIGHT = 1.0
    cfg.MODEL.MESH_HEAD.LAPLACIAN_LOSS_WEIGHT = 1.0
    cfg.MODEL.MESH_HEAD.DEPTH_LOSS_WEIGHT = 1.0
    cfg.MODEL.MESH_HEAD.SEMANTIC_LOSS_WEIGHT = 1.0
    # Init ico_sphere level (only for when voxel_on is false)
    cfg.MODEL.MESH_HEAD.ICO_SPHERE_LEVEL = -1
    # Mesh semantic label
    cfg.MODEL.MESH_HEAD.NUM_CLASSES = 5
    # Mesh projection focal length
    cfg.MODEL.MESH_HEAD.FOCAL_LENGTH = 2
    # Rendered image size
    cfg.MODEL.MESH_HEAD.IMAGE_SIZE = 512
    cfg.MODEL.MESH_HEAD.NUM_VERTICES = 1024

    # ------------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------------ #
    cfg.DATASETS = CN()
    cfg.DATASETS.NAME = "Sensat"
    cfg.DATASETS.DATA_DIR = "."
    cfg.DATASETS.TRAINSET = "train"
    cfg.DATASETS.VALSET = "val"
    cfg.DATASETS.TESTSET = "test"
    cfg.DATASETS.SAMPLES = 1000
    cfg.DATASETS.MESHING = ""
    cfg.DATASETS.DEPTH_SCALE = 100
    cfg.DATASETS.NORMALIZE_IMAGES = True
    cfg.DATASETS.SHUFFLE = True
    cfg.DATASETS.NUM_THREADS = 0
    cfg.DATASETS.NORMALIZE_DEPTH = True
    
    # ------------------------------------------------------------------------ #
    # Solver
    # ------------------------------------------------------------------------ #
    cfg.SOLVER = CN()
    cfg.SOLVER.LR_SCHEDULER_NAME = "constant"  # {'constant'}
    cfg.SOLVER.NUM_EPOCHS = 300
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.OPTIMIZER = "sgd"  # {'sgd', 'adam'}
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.BATCH_SIZE = 32
    cfg.SOLVER.BATCH_SIZE_EVAL = 8
    cfg.SOLVER.LOGGING_PERIOD = 50  # in iters
    cfg.SOLVER.GPU_ID = 0
    
    # ------------------------------------------------------------------------ #
    # Misc options
    # ------------------------------------------------------------------------ #
    # Directory where output files are written
    cfg.OUTPUT_DIR = "./output"

    return cfg