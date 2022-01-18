# the file to list the all necessary configurations for training/testing.

from fvcore.common.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
def get_sensat_cfg():
    cfg = CN()

    # ------------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------------ #
    cfg.DATASETS = CN()
    cfg.DATASETS.NAME = "sensat"
    cfg.DATASETS.DATA_DIR = "."
    cfg.DATASETS.TRAINSET = "train_birmingham"
    cfg.DATASETS.VALSET = "val_birmingham"
    cfg.DATASETS.TESTSET = "test_birmingham"
    cfg.DATASETS.NORMALIZE_IMAGES = True
    cfg.DATASETS.SHUFFLE = True
    cfg.DATASETS.BATCH_SIZE_TRAIN = 8
    cfg.DATASETS.BATCH_SIZE_VAL = 8
    cfg.DATASETS.BATCH_SIZE_TEST = 8
    cfg.DATASETS.NUM_WORKERS = 0
    
    # ------------------------------------------------------------------------ #
    # Solver
    # ------------------------------------------------------------------------ #
    cfg.SOLVER = CN()
    cfg.SOLVER.LR_SCHEDULER_NAME = "constant"  # {'constant'}
    cfg.SOLVER.NUM_EPOCHS = 300
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.OPTIMIZER = "sgd"  # {'sgd', 'adam'}
    cfg.SOLVER.MOMENTUM = 0.9

    return cfg