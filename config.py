import os

ROOT_DIR = os.environ["HOME"]
ROOT_DIR_LOCAL = '/state/share1'  # for local storgae

CURRENT_DIR = os.path.abspath(os.getcwd())

DATASET_FOLDER = "/datasets"

LOCAL_SERVER = 1
LIVIA_SERVER = 2

# _FER_DATASET_PATH = ROOT_DIR + '/Downloads/PhD/FER/datasets'    # Path for local server
_FER_DATASET_PATH = ROOT_DIR + DATASET_FOLDER                # Path for livia server

_BIOVID_DATASET_LOCAL_PATH = ROOT_DIR_LOCAL + DATASET_FOLDER      # Path for Biovid dataset local server

UNBC_PAIN_LABEL_PATH_ALL = _FER_DATASET_PATH + '/UNBCMcMaster/list_full.txt'
PAIN_LABEL_PATH_TRAIN = _FER_DATASET_PATH + '/UNBCMcMaster/sublist/1/list_train.txt'
PAIN_LABEL_PATH_VAL = _FER_DATASET_PATH + '/UNBCMcMaster/sublist/1/list_val.txt'
UNBC_PAIN_DB_PATH = _FER_DATASET_PATH + '/UNBCMcMaster/aligned_Images'

BIOVID_PATH = _BIOVID_DATASET_LOCAL_PATH + '/Biovid'

BIOVID_VIDEO_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/labels.txt'
BIOVID_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/image_labels.txt'
BIOVID_FULL_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/image_labels_full.txt'

BIOVID_SUBS_PATH = _BIOVID_DATASET_LOCAL_PATH + '/Biovid/sub_red_classes_img'

BIOVID_REDUCE_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/sub_two_labels.txt'


'''
this label path is by assigning every subject a class
'''  

BIOVID_SUBID_TO_SUBNAME_MAPPING = _FER_DATASET_PATH + '/Biovid/sub_mapping_N_classes.txt'

# BIOVID subjects folders
BIOVID_SUB_1_PATH = _FER_DATASET_PATH + '/Biovid/subjects/sub_1'
BIOVID_SUB_2_PATH = _FER_DATASET_PATH + '/Biovid/subjects/sub_2'
BIOVID_SUB_3_PATH = _FER_DATASET_PATH + '/Biovid/subjects/sub_3'

MCMASTER_PATH = _FER_DATASET_PATH + '/McMaster/aligned_Images'
MCMASTER_FULL_LABEL_PATH = _FER_DATASET_PATH + '/McMaster/mcmaster_list_full.txt'
MCMASTER_TWO_LABEL_PATH = _FER_DATASET_PATH + '/McMaster/mcmaster_two_cls_list.txt'

DATASET_FER = 'FER'
DATASET_RAF = 'RAF'
DATASET_AFEW = 'AFEW'
DATASET_AFFECTNET = 'AffectNet'
DATASET_JAFFE = 'JA'

DATASET_BIOVID = 'BIOVID'

TRAIN_SOURCE_AND_TARGET = 'Train both source and target'
TRAIN_ONLY_TARGET = 'Train only target'

PAIN_UNBC_MCMASTER = 'McMaster'
PAIN_BIOVID = 'Biovid'

# ----------------- Distance Measures ----------

MMD_SIMILARITY = 1
COSINE_SIMILARITY = 2
KL_DIVERGENCE = 3

# ------------------------------------------

WEIGHTS_FOLDER = "WeightFiles"
BIOVID_N_SRC_WEIGHT_FILE = 'WeightFiles/lab_srcs78_cl77_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only'
ALL_SOURCES_FOLDER = "AllSources"

# --- Define your Comet key here
COMET_API_KEY = "YOUR_COMET_API_KEY"
COMET_WORKSPACE = "YOUR_COMET_WORKSPACE"
COMET_LOG_CODE = True
COMET_DISABLED = False
COMET_PROJECT_NAME = "YOUR_COMET_PROJECT_NAME"