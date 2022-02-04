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
    cfg.MODEL.CHANNELS = 3
    cfg.MODEL.SEMANTIC = False

    # ------------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------------ #
    cfg.MODEL.RESUME = False
    cfg.MODEL.RESUME_MODEL = ""

    # ------------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------------ #
    cfg.MODEL.CHECKPOINT = "./checkpoints"  # path to checkpoint

    # ------------------------------------------------------------------------ #
    # Deeplab Segmentation
    # ------------------------------------------------------------------------ #
    cfg.MODEL.DEEPLAB = CN()
    cfg.MODEL.DEEPLAB.LR = 0.01
    cfg.MODEL.DEEPLAB.MOMENTUM = 0.9
    cfg.MODEL.DEEPLAB.WEIGHT_DECAY = 1e-4
    cfg.MODEL.DEEPLAB.SCHEDULER = 50
    cfg.MODEL.DEEPLAB.SCHEDULER_STEP_SIZE = 50
    cfg.MODEL.DEEPLAB.SCHEDULER_GAMMA = 0.3
    cfg.MODEL.DEEPLAB.NUM_EPOCHS = 300
    cfg.MODEL.DEEPLAB.CLASS_WEIGHTED = False
    cfg.MODEL.DEEPLAB.CLASS_WEIGHT = [1, 1, 1, 1, 1]
    cfg.MODEL.DEEPLAB.LOSS = "cross_entropy"
    cfg.MODEL.DEEPLAB.NUM_CLASSES = 5
    cfg.MODEL.DEEPLAB.PRETRAIN = False

    # ------------------------------------------------------------------------ #
    # Mesh Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.MESH_HEAD = CN()
    cfg.MODEL.MESH_HEAD.NAME = "VoxMeshHead"
    cfg.MODEL.MESH_HEAD.SEM_PRETRAIN_MODEL_PATH = ""
    cfg.MODEL.MESH_HEAD.RESNET_PRETRAIN = False
    # Numer of stages
    cfg.MODEL.MESH_HEAD.NUM_STAGES = 1
    cfg.MODEL.MESH_HEAD.NUM_GRAPH_CONVS = 1  # per stage
    cfg.MODEL.MESH_HEAD.GRAPH_CONV_DIM = 256
    cfg.MODEL.MESH_HEAD.GRAPH_CONV_INIT = "normal"
    cfg.MODEL.MESH_HEAD.GRAPH_CONV_SEMANTIC = True
    cfg.MODEL.MESH_HEAD.FREEZE_CLASSIFIER = True
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
    # Deform threshold
    cfg.MODEL.MESH_HEAD.OFFSET_THRESHOLD = 1.0

    # ------------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------------ #
    cfg.DATASETS = CN()
    cfg.DATASETS.NAME = "Sensat"
    cfg.DATASETS.DATA_DIR = "."
    cfg.DATASETS.TRAINSET = "train"
    cfg.DATASETS.VALSET = "val"
    cfg.DATASETS.TESTSET = "test"
    cfg.DATASETS.SAMPLES = [1000]
    cfg.DATASETS.MESHING = ""
    cfg.DATASETS.DEPTH_SCALE = 100
    cfg.DATASETS.SHUFFLE = True
    cfg.DATASETS.NUM_THREADS = 0
    cfg.DATASETS.NORMALIZE_IMAGES = True
    cfg.DATASETS.NORMALIZE_MESH = True
    cfg.DATASETS.SIZE = 0

    # ------------------------------------------------------------------------ #
    # Solver
    # ------------------------------------------------------------------------ #
    cfg.SOLVER = CN()
    cfg.SOLVER.SCHEDULER = "ReduceLROnPlateau" # {'ReduceLROnPlateau','StepLR'}
    cfg.SOLVER.SCHEDULER_STEP_SIZE = 50
    cfg.SOLVER.SCHEDULER_GAMMA = 0.5
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