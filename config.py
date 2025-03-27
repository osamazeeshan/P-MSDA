import os
# import argparse

ROOT_DIR = os.environ["HOME"]
ROOT_DIR_LOCAL = '/state/share1'  # for local storgae

CURRENT_DIR = os.path.abspath(os.getcwd())


DATASET_FOLDER = "/datasets"

LOCAL_SERVER = 1
LIVIA_SERVER = 2

# _FER_DATASET_PATH = ROOT_DIR + '/Downloads/PhD/FER/datasets'    # Path for local server
_FER_DATASET_PATH = ROOT_DIR + DATASET_FOLDER                # Path for livia server

_BIOVID_DATASET_LOCAL_PATH = ROOT_DIR_LOCAL + DATASET_FOLDER      # Path for Biovid dataset local server

RAF_DB_PATH = _FER_DATASET_PATH + '/RAF/basic/Image/aligned_train'
RAF_DB_TEST_PATH = _FER_DATASET_PATH + '/RAF/basic/Image/aligned_test'
FER_DB_TRAIN_PATH = _FER_DATASET_PATH + '/FER2013/train'
FER_DB_TEST_PATH = _FER_DATASET_PATH + '/FER2013/test'
AFEW_DB_PATH = _FER_DATASET_PATH + '/AFEW/Faces/Train/ver-2'
AFEW_TEST_DB_PATH = _FER_DATASET_PATH + '/AFEW/Faces/Valid'

AFFECTNET_TRAIN_PATH = _FER_DATASET_PATH + '/AffectNet/manually_annotated_split/train'
AFFECTNET_TEST_PATH = _FER_DATASET_PATH + '/AffectNet/manually_annotated_split/test'

# perform adaptation on each subject
# JAFFE contains data of total 10 subjects with 3-4 images in each class 
JAFFE_SUB_1_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_1/train'
JAFFE_SUB_1_TEST_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_1/test'

JAFFE_SUB_2_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_2/train'
JAFFE_SUB_2_TEST_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_2/test'

JAFFE_SUB_3_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_3/train'
JAFFE_SUB_3_TEST_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_3/test'

JAFFE_SUB_4_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_4/train'
JAFFE_SUB_4_TEST_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_4/test'

JAFFE_SUB_5_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_5/train'
JAFFE_SUB_5_TEST_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_5/test'

JAFFE_SUB_6_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_6/train'
JAFFE_SUB_6_TEST_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_6/test'

JAFFE_SUB_7_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_7/train'
JAFFE_SUB_7_TEST_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_7/test'

JAFFE_SUB_8_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_8/train'
JAFFE_SUB_8_TEST_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_8/test'

JAFFE_SUB_9_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_9/train'
JAFFE_SUB_9_TEST_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_9/test'

JAFFE_SUB_10_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_10/train'
JAFFE_SUB_10_TEST_PATH = _FER_DATASET_PATH + '/subject_jaffe_aug/sub_10/test'


# sources dictionary
# src_train_datasets = [JAFFE_SUB_1_PATH, JAFFE_SUB_2_PATH]
# src_test_datasets = [JAFFE_SUB_1_TEST_PATH, JAFFE_SUB_2_TEST_PATH]

# src_train_datasets = [JAFFE_SUB_1_PATH, JAFFE_SUB_2_PATH, JAFFE_SUB_4_PATH, JAFFE_SUB_5_PATH]
# src_test_datasets = [JAFFE_SUB_1_TEST_PATH, JAFFE_SUB_2_TEST_PATH, JAFFE_SUB_4_TEST_PATH, JAFFE_SUB_5_TEST_PATH]

src_train_datasets = [JAFFE_SUB_1_PATH, JAFFE_SUB_3_PATH, JAFFE_SUB_4_PATH, JAFFE_SUB_5_PATH, JAFFE_SUB_6_PATH, JAFFE_SUB_7_PATH]
src_test_datasets = [JAFFE_SUB_1_TEST_PATH, JAFFE_SUB_3_TEST_PATH, JAFFE_SUB_4_TEST_PATH, JAFFE_SUB_5_TEST_PATH, JAFFE_SUB_6_TEST_PATH, JAFFE_SUB_7_TEST_PATH]

# src_train_datasets = [JAFFE_SUB_1_PATH, JAFFE_SUB_2_PATH, JAFFE_SUB_4_PATH, JAFFE_SUB_5_PATH, JAFFE_SUB_6_PATH, JAFFE_SUB_7_PATH, JAFFE_SUB_8_PATH, JAFFE_SUB_9_PATH, JAFFE_SUB_10_PATH]
# src_test_datasets = [JAFFE_SUB_1_TEST_PATH, JAFFE_SUB_2_TEST_PATH, JAFFE_SUB_4_TEST_PATH, JAFFE_SUB_5_TEST_PATH, JAFFE_SUB_6_TEST_PATH, JAFFE_SUB_7_TEST_PATH, JAFFE_SUB_8_TEST_PATH, JAFFE_SUB_9_TEST_PATH, JAFFE_SUB_10_TEST_PATH]

# src_train_datasets = [AFFECTNET_TRAIN_PATH, FER_DB_TRAIN_PATH]
# src_test_datasets = [AFFECTNET_TEST_PATH, FER_DB_TEST_PATH]


UNBC_PAIN_LABEL_PATH_ALL = _FER_DATASET_PATH + '/UNBCMcMaster/list_full.txt'
PAIN_LABEL_PATH_TRAIN = _FER_DATASET_PATH + '/UNBCMcMaster/sublist/1/list_train.txt'
PAIN_LABEL_PATH_VAL = _FER_DATASET_PATH + '/UNBCMcMaster/sublist/1/list_val.txt'
UNBC_PAIN_DB_PATH = _FER_DATASET_PATH + '/UNBCMcMaster/aligned_Images'

# BIOVID_PATH = _FER_DATASET_PATH + '/Biovid'

 # ------------------ ARGUMENTS -------------------- #
# arg_parser = argparse.ArgumentParser(description='Train a network on FER')
# arg_parser.add_argument('--src_train_datasets_path', type=str, help='Dataset path')
# arg_parser.add_argument('--src_test_datasets_path', type=str, help='Dataset path')
# arg_parser.add_argument('--pain_db_root_path', type=str, help='Dataset path')

# args = arg_parser.parse_args()
 # ------------------ ----------------- -------------------- #

BIOVID_PATH = _BIOVID_DATASET_LOCAL_PATH + '/Biovid'

# BIOVID_PATH = args.src_train_datasets_path

BIOVID_VIDEO_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/labels.txt'
BIOVID_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/image_labels.txt'
BIOVID_FULL_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/image_labels_full.txt'

# BIOVID_SUBS_PATH = _FER_DATASET_PATH + '/Biovid/face_images'
# BIOVID_SUBS_PATH = _FER_DATASET_PATH + '/Biovid/face_images_m'

# Reduce classes
# BIOVID_SUBS_PATH = _FER_DATASET_PATH + '/Biovid/sub_img_red_classes'

BIOVID_SUBS_PATH = _BIOVID_DATASET_LOCAL_PATH + '/Biovid/sub_red_classes_img'
# BIOVID_SUBS_PATH = args.pain_db_root_path

# BIOVID_REDUCE_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/sub_red_labels.txt'

BIOVID_REDUCE_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/sub_two_labels.txt'




'''
this label path is by assigning every subject a class
'''  
# BIOVID_REDUCE_LABEL_PATH = _FER_DATASET_PATH + '/Biovid/sub_N_source_classes.txt'


BIOVID_SUBID_TO_SUBNAME_MAPPING = _FER_DATASET_PATH + '/Biovid/sub_mapping_N_classes.txt'

# BIOVID subjects folders
BIOVID_SUB_1_PATH = _FER_DATASET_PATH + '/Biovid/subjects/sub_1'
BIOVID_SUB_2_PATH = _FER_DATASET_PATH + '/Biovid/subjects/sub_2'
BIOVID_SUB_3_PATH = _FER_DATASET_PATH + '/Biovid/subjects/sub_3'


MCMASTER_PATH = _FER_DATASET_PATH + '/McMaster/aligned_Images'
MCMASTER_FULL_LABEL_PATH = _FER_DATASET_PATH + '/McMaster/mcmaster_list_full.txt'

MCMASTER_TWO_LABEL_PATH = _FER_DATASET_PATH + '/McMaster/mcmaster_two_cls_list.txt'
# BIOVID_LABEL_PATH_TRAIN = _FER_DATASET_PATH + '/Biovid/sublist/1/list_train.txt'
# BIOVIDLABEL_PATH_VAL = _FER_DATASET_PATH + '/Biovid/sublist/1/list_val.txt'
# BIOVID_DB_PATH = _FER_DATASET_PATH + '/Biovid/face_images'

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


OFFICE_31_PATH = '/home/osamazeeshan/Downloads/PhD/office-31/train/amazon'
FER_MULTI_DOMAINS_PATH = '/home/osamazeeshan/Downloads/PhD/FER/datasets/facial-exp'

# ----------------- Distance Measures ----------

MMD_SIMILARITY = 1
COSINE_SIMILARITY = 2
KL_DIVERGENCE = 3

# ------------------------------------------

WEIGHTS_FOLDER = "WeightFiles"
BIOVID_N_SRC_WEIGHT_FILE = 'WeightFiles/lab_srcs78_cl77_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only'
ALL_SOURCES_FOLDER = "AllSources"

COMET_API_KEY = "eow2bmNwSPBKrx657Qfx43lW7"
COMET_WORKSPACE = "osamazeeshan"
COMET_LOG_CODE = True
COMET_DISABLED = False
COMET_PROJECT_NAME = "source-selection-self-paced-msda"