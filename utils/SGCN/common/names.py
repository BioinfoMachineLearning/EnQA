TMP = 'tmp'
LOG = 'log'
INFO = 'info'

TARGET_FILE_NAME = 'target.pdb'
MODEL_FILE_NAME = 'model.pdb'

X_FILE_NAME = 'x.txt'
X_RES_FILE_NAME = 'x_res.txt'
Y_FILE_NAME = 'y.txt'
Y_GLOBAL_FILE_NAME = 'y_global.txt'
ADJ_B_NAME = 'adj_b.txt'
ADJ_C_NAME = 'adj_c.txt'
ADJ_RES_NAME = 'adj_res.txt'
AGGR_MASK_NAME = 'aggr.txt'
COVALENT_TYPES_NAME = 'covalent_types.txt'
SEQ_SEP_NAME = 'sequence_separation.txt'
SH_NAME = 'sh.npy'

MODEL_IMPORTANCES_FILE_NAME = 'cad_coeffs.txt'
INCOMPLETENESS_FILE_NAME = 'incompleteness.txt'

SUPPORTING_OBJECTS = [
    TMP,
    LOG,
    INFO,
    TARGET_FILE_NAME,
    MODEL_IMPORTANCES_FILE_NAME,
    INCOMPLETENESS_FILE_NAME]

NEAR_NATIVE_NAME_PATTERN = 'nlb_decoy'
CHECKPOINT_EPOCH_PREFIX = 'checkpoint_epoch'

# Graph dict fields
TARGET_NAME_FIELD = 'target_name'
MODEL_NAME_FIELD = 'model_name'
ONE_HOT_FIELD = 'one_hot'
FEATURES_FIELD = 'features'
GEMME_FEATURES_FIELD = 'gemme_features'
Y_FIELD = 'y'
Y_GLOBAL_FIELD = 'y_global'
ADJ_B_FIELD = 'adj_b'
ADJ_C_FIELD = 'adj_c'
ADJ_RES_FIELD = 'adj_res'
AGGR_MASK_FIELD = 'aggr_mask'
SH_FIELD = 'sh'

# Model state dict fields
MODEL_DESCRIPTION = 'model_description'
MODEL_STATE = 'model_state'
OPTIMIZER_STATE = 'optimizer_state'
TRAIN_LOSS_HISTORY = 'train_loss_history'
TRAIN_CORR_HISTORY = 'train_corr_history'
TRAIN_TIME_HISTORY = 'train_time_history'
EPOCH = 'epoch'

EPSILON = 1e-15
X_NORM_IDX = [0, 2]
