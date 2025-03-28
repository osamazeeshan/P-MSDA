# import comet_ml at the top of your file
from comet_ml import Experiment
import argparse
import os, sys

# os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import torch
from torchmetrics import Accuracy
from tqdm import tqdm, trange
from torch import nn
import random
# import cv2
from scipy import stats
import time
from numpy.linalg import norm

# from thop import profile
# from ptflops import get_model_complexity_info
import re
from collections import Counter
import copy

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils.common import *
from utils.distance import compute_distance_matrix
from methods.clusters import *
# from utils.visualize import *

from datasets.base_dataset import BaseDataset
from datasets.dataset import TragetRestartableIterator
from models.resnet_model_modified import ResNet50Fc, ResNet18Fc
from losses.coral_loss import CORAL_loss
from losses.mmd_loss import MMD_loss
import torch.nn.functional as F

from models.resnet_fer_modified import *
from pathlib import Path

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score

from utils.tsne_variation import *

from methods.create_prototypes import create_target_pl_dicts
from methods.clusters import create_relv_src_clus_dbscan, create_relv_src_clusters, create_relv_src_dic, create_relv_src_clus_cent
from networks.transfer_net import TransferNet as TN 

from torch.utils.data import TensorDataset

from utils.reproducibility import set_default_seed, set_to_deterministic, _get_current_seed
from sklearn.metrics import pairwise_distances_argmin_min

from scipy.spatial.distance import cdist

import config
import itertools

from utils.cometml import comet_init, set_comet_exp_name

# from losses.supcontrast_loss import SupConLoss


# Create an experiment with your api key
# experiment = Experiment(
#     api_key="eow2bmNwSPBKrx657Qfx43lW7",
#     project_name="source-selection-self-paced-msda",
#     workspace="osamazeeshan",
#     log_code=True,
#     disabled=False,
# )


experiment = comet_init(config.COMET_PROJECT_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transfer_loss = 'mmd'
learning_rate = 0.0001
# n_class = 7


prev_srcs_loader_dic = []

# n_class = 2 # for pain Biovid
# n_class = 77 # to train a classifier with N source subject classes

set_default_seed()
set_to_deterministic()

def calculate_src_tar_dist(source_list_name, target_subject, eliminate_list, transfer_model, lamb_threshold, target_file_path, dist_measure):
    src_tar_dist_dic = {}
    for src_sub in source_list_name:
        if src_sub in eliminate_list:
            continue
        subject_list = [src_sub]
        subject_list.append(target_subject)
        print(subject_list)

        subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.BIOVID_REDUCE_LABEL_PATH, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
        srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

        # subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.MCMASTER_FULL_LABEL_PATH, subject_list, target_subject, args.n_class, args.oracle_setting)
        # srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

        # srcs_file_name = 'mcmaster_list_full.txt'   # to train model for McMaster to use as a pre-trained model
        combine_srcs_loader, combine_srcs_val_loader, combine_srcs_test_loader = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, srcs_file_name), None, args.batch_size, phase='src')
        tar_loader, tar_val_loader, tar_test_loader  = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, tar_file_name), None, args.batch_size, phase='tar')

        # calculate Source and Target subject distance
        dist = measure_srcs_tar_dist(transfer_model, combine_srcs_loader, tar_loader, dist_measure, args.batch_size)
        src_tar_dist_dic[src_sub] = dist
        # dist_array.append(dist)
    
    src_tar_dist_dic = dict(sorted(src_tar_dist_dic.items(), key=lambda x:x[1], reverse=True))
    src_sub_normalized = normalize_data(list(src_tar_dist_dic.values())) # normalize the distance between 0 and 1 to apply threshold 
    src_tar_dist_dic = dict(itertools.islice(src_tar_dist_dic.items(), np.sum(src_sub_normalized > lamb_threshold))) # select baased on the threshold
    
    # src_tar_dist_dic = dict(itertools.islice(src_tar_dist_dic.items(), 10)) # select based on the closest topk=10
    # torch.argmax(torch.nn.functional.softmax(src_tar_dist_dic, dim=1), dim=1).cpu().numpy()
    # print(select_topk_dist.keys())
    print(src_tar_dist_dic)
    return src_tar_dist_dic

def calculate_src_sampl_tar_dist(source_list_name, target_subject, transfer_model, target_file_path, dist_measure):
    src_tar_dist_dic = {}
    # for src_sub in source_list_name:
    #     if src_sub in eliminate_list:
    #         continue
    subject_list = source_list_name
    # subject_list.append(target_subject)
    print(subject_list)

    subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.BIOVID_REDUCE_LABEL_PATH, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
    srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

    # subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.MCMASTER_FULL_LABEL_PATH, subject_list, target_subject, args.n_class, args.oracle_setting)
    # srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

    # srcs_file_name = 'mcmaster_list_full.txt'   # to train model for McMaster to use as a pre-trained model
    combine_srcs_loader, combine_srcs_val_loader, combine_srcs_test_loader = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, srcs_file_name), None, 100, phase='src')
    tar_loader, tar_val_loader, tar_test_loader  = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, tar_file_name), None, 100, phase='tar')

    # calculate dis of every source sample with tar
    close_src_sample_tar_numpy = 'closest_src_samples_tar_'+target_subject+'.npz'
    if not Path(close_src_sample_tar_numpy).is_file():
        measure_src_samples_tar_dist(transfer_model, combine_srcs_loader, tar_loader, dist_measure, close_src_sample_tar_numpy, args.batch_size)
    
    closest_loaded_arrays = np.load(close_src_sample_tar_numpy)
    return closest_loaded_arrays

def get_closest_src_sample_tar(closest_loaded_arrays, relv_sample_count):
    # Loading the arrays
    
    closest_src_samples = closest_loaded_arrays['closest_src_samples']
    closest_src_labels = closest_loaded_arrays['closest_src_labels']
    closest_src_dists = closest_loaded_arrays['closest_src_dists']

    sorted_indices = np.flip(np.argsort(closest_src_dists))
    # get the sorted top N previous src samples indices
    relv_indices = sorted_indices[:relv_sample_count].tolist()
    closest_src_samples_arr = np.array(closest_src_samples)
    closest_src_samples_arr = closest_src_samples_arr[relv_indices]
    closest_src_labels_arr = np.array(closest_src_labels)
    closest_src_labels_arr = closest_src_labels_arr[relv_indices]


    return closest_src_samples_arr, closest_src_labels_arr


def get_top_closest_target_subject(transfer_model, target_file_path, tar_file_name):
    tar_loader, tar_val_loader, tar_test_loader  = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, tar_file_name), None, args.batch_size, phase='tar')
    transfer_model, _ = initialize_model(args, os.path.join(config.CURRENT_DIR, config.BIOVID_N_SRC_WEIGHT_FILE), True, args.n_class_N_src, pretrained_model_name = None) # pretrained_model_name = args.pretrained_model

    # target evaluation on trianed source model
    top_srcs_train = test(transfer_model, tar_loader, args.batch_size, True, args.is_pain_dataset)

    # get subject name from subject Ids
    top_subname_dic = map_subid_to_subname(config.BIOVID_SUBID_TO_SUBNAME_MAPPING, top_srcs_train)
    print(top_subname_dic)

    return top_subname_dic
    
def main(args):

    '''
    BIOVID DATSET:
        - write text file for the numbers of subjects for sources and target
        - create loader for source and target 
        - create src and traget file name
    '''
    
    '''
    BioVid Random target subjects
    '''
    source_list_name = ['082208_w_45', '081714_m_36', '112610_w_60', '101908_m_61', '071709_w_23','082014_w_24', '110810_m_62', '080209_w_26', '101916_m_40', '110614_m_42',
    '101814_m_58', '112016_m_25', '071313_m_41', '102514_w_40', '100514_w_51', '101114_w_37', '100509_w_43', '082315_w_60', '112310_m_20', '120614_w_61', 
    '092714_m_64', '101514_w_36', '092813_w_24', '102414_w_58', '102309_m_61', '081617_m_27', '080609_w_27', '083114_w_55', '111313_m_64', '071614_m_20', 
    '101309_m_48', '071911_w_24', '102316_w_50', '100417_m_44', '083013_w_47', '083009_w_42', '080714_m_23', '101809_m_59', '082909_m_47', '101209_w_61', 
    '092014_m_56', '072414_m_23', '101015_w_43', '112909_w_20', '111609_m_65', '100117_w_36', '111409_w_63', '080709_m_24', '072714_m_23', '112914_w_51', 
    '120514_w_56', '083109_m_60', '110909_m_29', '091814_m_37', '071814_w_23', '092509_w_51', '112809_w_23', '100214_m_50', '102214_w_36', '082714_m_22', 
    '082109_m_53', '092808_m_51', '080309_m_29', '102008_w_22', '111914_w_63', '082809_m_26', '072514_m_27', '082814_w_46', '072609_w_23', '101216_m_40', 
    '091914_m_46', '100914_m_39', '112209_m_51', '092514_m_50', '092009_m_54', '082414_m_64', '080614_m_24']

    # source_list_name = ['082208_w_45', '081714_m_36', '082109_m_53']

    # --- Target subjects
    target_random_list = ['081014_w_27','101609_m_36','112009_w_43','091809_w_43','071309_w_21','073114_m_25','080314_w_25','073109_w_28','100909_w_65','081609_w_40']

    '''
    McMaster subjects
    '''
    unbc_source_list_name = ['047-jl047', '095-tv095', '048-aa048', '049-bm049', '097-gf097', '052-dr052',  '059-fn059', '124-dn124',   '042-ll042',   
    '106-nm106', '066-mg066','096-bg096', '043-jh043', '101-mg101', '103-jk103', '108-th108', '080-bn080', '120-kz120', '064-ak064', '092-ch092']
    unbc_target_subject_list = ['107-hs107', '109-ib109', '121-vw121', '123-jh123', '115-jy115'] 
    
    target_subject = target_random_list[args.tar_subject] # 081609_w_40 # 101609_m_36 # 073109_w_28
    print("== Target Subject: ",target_subject)

    # ---------------------------------- ----------------------------------- #

    transfer_model, optimizer = initialize_model(args, None, args.load_source, args.n_class, pretrained_model_name = None) # pretrained_model_name = args.pretrained_model
    # transfer_model=None
    # optimizer=None
    test_model_name = None

    eliminate_list = []
    selected_sub = 0
    
    count_subs = 1

    relv_feat_arr = []
    relv_src_data_arr = [] 
    relv_src_label_arr = []
    prev_relv_samples_struc = None

    _BEST_VAL_ACC = 0
    is_ignore_model = False 
    prev_src_sub_model = ''

    # comet create experiment name
    set_comet_exp_name(experiment, args.top_s, args.source_combined, len(source_list_name), target_subject)
    target_file_path, target_weight_path, timestamp = create_target_folders(config.CURRENT_DIR, args.weights_folder, target_subject, args.top_timestamp if args.target_evaluation_only else None)


    # Selection of top source subjects w.r.t each target using N classifier 
    if args.train_N_source_classes and selected_sub == 0:
        subject_list = source_list_name
        subject_list.append(target_subject)

        subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.BIOVID_REDUCE_LABEL_PATH, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
        srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

        top_N_srcs_tar = get_top_closest_target_subject(transfer_model, target_file_path, tar_file_name)

    transfer_model, optimizer = initialize_model(args, None, args.load_source, args.n_class, pretrained_model_name = None)
    counter_based_N_classifer = 0
    if args.train_source or args.load_source:
        subject_list = source_list_name
        subject_list.append(target_subject)

        subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.BIOVID_REDUCE_LABEL_PATH, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
        srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)
        transfer_model, test_model_name, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model = domain_adaptation(srcs_file_name, tar_file_name, transfer_model, optimizer, target_subject, count_subs, test_model_name, target_file_path, target_weight_path, timestamp, args.dist_measure, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model)
    
    if args.train_w_src_sm:
        source_model_name = 'WeightFiles/lab_srcs78_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only'
        transfer_model, optimizer = initialize_model(args, source_model_name, True, args.n_class, pretrained_model_name = None) # pretrained_model_name = args.pretrained_model
        
    relv_sample_count = 2000
    closest_src_samples_arr = [] 
    closest_src_labels_arr = []
    if args.train_w_src_sm:
        subject_list = source_list_name
        subject_list.append(target_subject)
        # source_model_name = 'WeightFiles/lab_srcs78_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only'
        # transfer_model, optimizer = initialize_model(args, source_model_name, True, args.n_class, pretrained_model_name = None) # pretrained_model_name = args.pretrained_model
        if not args.train_w_rand_src_sm:
            closest_loaded_arrays = dist_measure_dic = calculate_src_sampl_tar_dist(source_list_name, target_subject, transfer_model, target_file_path, args.dist_measure)
        while relv_sample_count < args.top_rev_src_sam:
            if not args.train_w_rand_src_sm:
                closest_src_samples_arr, closest_src_labels_arr = get_closest_src_sample_tar(closest_loaded_arrays, relv_sample_count)

            subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.BIOVID_REDUCE_LABEL_PATH, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
            srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

            transfer_model, optimizer = initialize_model(args, None, args.load_source, args.n_class, pretrained_model_name = None)
            transfer_model, test_model_name, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, _, _, _, _ = domain_adaptation(srcs_file_name, None, tar_file_name, transfer_model, optimizer, target_subject, relv_sample_count, test_model_name, target_file_path, target_weight_path, timestamp, args.dist_measure, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, None, None, None, None, closest_src_samples_arr, closest_src_labels_arr)
            relv_sample_count = relv_sample_count + 2000

    elif not args.target_evaluation_only:
        while selected_sub < args.top_s:
            if args.train_N_source_classes and args.train_with_dist_measure:
                if target_subject in source_list_name:
                    source_list_name.remove(target_subject)

                dist_measure_dic = calculate_src_tar_dist(source_list_name, target_subject, eliminate_list, transfer_model, args.cs_threshold, target_file_path, args.dist_measure)
                # dist_measure_dic = {'083109_m_60': 0.6030410408973694, '102309_m_61': 0.6049227381861487, '072414_m_23': 0.6055345017176408}
                N_dist_dic = top_N_srcs_tar

                if N_dist_dic:
                    pick_top_N_src_sb = min(3, len(N_dist_dic))
                    src_subjects = list(set(dist_measure_dic).intersection(dict(list(N_dist_dic.items())[:pick_top_N_src_sb])))     # get common subjects from dist measure and N-Classifier
                else:
                    src_subjects = []

                if len(src_subjects) > 0:
                    print("\n ---------------------- -------------------- -----------------\n")
                    print("Common Source Subjects B/W Dist Measure and N-Classifier): ", src_subjects)
                    print("\n ---------------------- -------------------- -----------------\n")
                    top_N_srcs_tar = {k: v for k, v in top_N_srcs_tar.items() if k not in src_subjects}    # remove key/value pair from N-classifier dic 
                else:
                    pick_top_N_src_sb = min(3, len(N_dist_dic) if N_dist_dic else len(dist_measure_dic))
                    src_subjects = list(N_dist_dic.keys())[:pick_top_N_src_sb] if N_dist_dic else list(dist_measure_dic.keys())[:pick_top_N_src_sb]

                    top_N_srcs_tar = {k: v for k, v in top_N_srcs_tar.items() if k not in src_subjects}    # remove key/value pair from N-classifier dic 
                    print("\n ---------------------- --------------------\n")
                    print("No Common Source Subjects")
                    print("Selecting subjects: ", src_subjects)
                    print("\n ---------------------- --------------------\n")

            elif args.train_N_source_classes and selected_sub == 0 :
                source_list_name.remove(target_subject)
                src_subjects = list(top_N_srcs_tar.keys())
                counter_based_N_classifer = len(src_subjects)
                print("counter_based_N_classifer: ", counter_based_N_classifer)
                # src_tar_dist_dic = {'082109_m_53': 0.604693013888139}
            else:
                src_tar_dist_dic = calculate_src_tar_dist(source_list_name, target_subject, eliminate_list, transfer_model, args.cs_threshold, target_file_path, args.dist_measure)
                src_subjects = list(src_tar_dist_dic.keys())
    
            for src_key in src_subjects:
                eliminate_list.append(src_key)

            selected_sub = selected_sub + len(src_subjects)

            '''
                if : combine selected source subjects and adapt to target
                else : selected source subjects will be adpted individually with target  
            '''
            if args.source_combined:
                subject_list = src_subjects
                subject_list.append(target_subject)

                subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.BIOVID_REDUCE_LABEL_PATH, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
                srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)
                transfer_model, test_model_name, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, _, _, _, _ = domain_adaptation(srcs_file_name, None, tar_file_name, transfer_model, optimizer, target_subject, count_subs, test_model_name, target_file_path, target_weight_path, timestamp, args.dist_measure, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model) 
            else:
                for src in src_subjects:
                    # subject_list = [src]
                    '''
                        P-MSDA: using two source subject loaders.
                        First loader contains all the previous subjects, Second loader contain only the newly added source subject
                        - In case of first sub, put same sub in both loaders
                    ''' 
                    # prev_subject_list = None
                    if args.accumulate_prev_source_subs:
                        print("\n *** Accumulating previously adapted source subjects in the first loader *** \n")
                        if count_subs > 1:
                            prev_subject_list.remove(target_subject)
                            prev_subject_list.append(src)
                        else:
                            prev_subject_list = [src]
                        prev_subject_list.append(target_subject)

                    subject_list = [src]
                    subject_list.append(target_subject)

                     # generating file for previous source sub loader 
                    if args.accumulate_prev_source_subs:
                        prev_subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.BIOVID_REDUCE_LABEL_PATH, prev_subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
                        prev_srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(prev_subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

                    subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.BIOVID_REDUCE_LABEL_PATH, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
                    srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)
                    transfer_model, test_model_name, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, relv_feat_arr, relv_src_data_arr, relv_src_label_arr, prev_relv_samples_struc = domain_adaptation(srcs_file_name, prev_srcs_file_name, tar_file_name, transfer_model, optimizer, target_subject, count_subs, test_model_name, target_file_path, target_weight_path, timestamp, args.dist_measure, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, relv_feat_arr, relv_src_data_arr, relv_src_label_arr, prev_relv_samples_struc, closest_src_samples_arr, closest_src_labels_arr)
                    count_subs = count_subs + 1

        print("\n ------------------ ------------------\n")
        print("Source selection based on N classifier: ", counter_based_N_classifer)
        print("\n ------------------ ------------------\n")

        print("\n ------------------ ------------------\n")
        print("Source selection based on threshold: ", counter_based_N_classifer - count_subs)
        print("\n ------------------ ------------------\n")
    else:
        subject_list = [args.top_src_sub]
        subject_list.append(target_subject)

        # subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.BIOVID_REDUCE_LABEL_PATH, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
        srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)
        tar_file_name='lab_srcs3_083013w47_112809w23_tar_081609w40.txt'
        srcs_file_name='lab_srcs3_083013w47_112809w23_only.txt'
        transfer_model, test_model_name, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, _, _, _, _ = domain_adaptation(srcs_file_name, None, tar_file_name, transfer_model, optimizer, target_subject, count_subs, test_model_name, target_file_path, target_weight_path, timestamp, args.dist_measure, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model)

def domain_adaptation(srcs_file_name, prev_srcs_file_name, tar_file_name, transfer_model, optimizer, 
                      target_subject, count_subs, test_model_name, target_file_path, target_weight_path, 
                      timestamp, dist_measure, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, relv_feat_arr, 
                      relv_src_data_arr, relv_src_label_arr, prev_relv_samples_struc, closest_src_samples_arr, 
                      closest_src_labels_arr):
    
    combine_srcs_loader, combine_srcs_val_loader, combine_srcs_test_loader = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, srcs_file_name), None, args.batch_size, phase='src')
    tar_loader, tar_val_loader, tar_test_loader  = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, tar_file_name), None, args.batch_size, phase='tar')

    source_model_name = srcs_file_name.split('.')[0]
    tar_model_name = tar_file_name.split('.')[0] + "_oracle" if args.oracle_setting else tar_file_name.split('.')[0] 
    lamb = 0.5 # weight for transfer loss, it is a hyperparameter that needs to be tuned

    relv_sample_count = 2000
    # *** *** Selection of relevant source samples --- ****
    if args.accumulate_prev_source_subs:
        # if len(relv_src_data_arr) > 0 and len(relv_src_label_arr):
        if prev_relv_samples_struc:
            relv_src_data_arr = [point[0] for point in prev_relv_samples_struc if point[0] is not None]
            relv_src_label_arr = [point[2] for point in prev_relv_samples_struc if point[2] is not None]

            relv_prev_src_subs = TensorDataset(torch.tensor(np.array(relv_src_data_arr)), torch.tensor(relv_src_label_arr)) 
            prev_srcs_loader, prev_srcs_val_loader = BaseDataset.load_target_data(relv_prev_src_subs, args.batch_size, split=False)
        else:
            prev_srcs_loader, prev_srcs_val_loader = combine_srcs_loader, combine_srcs_val_loader,

    if args.train_w_src_sm:
        if len(closest_src_samples_arr) > 0 and len(closest_src_labels_arr) > 0:
            relv_prev_src_subs = TensorDataset(torch.tensor(np.array(closest_src_samples_arr)), torch.tensor(closest_src_labels_arr)) 
            prev_srcs_loader, prev_srcs_val_loader = BaseDataset.load_target_data(relv_prev_src_subs, args.batch_size, split=False)
        elif args.train_w_rand_src_sm:
            prev_srcs_loader = BaseDataset.generate_random_src_samples(combine_srcs_loader, count_subs)
            prev_srcs_val_loader = BaseDataset.generate_random_src_samples(combine_srcs_val_loader, 300)
        tar_model_name = tar_model_name + '_' + str(count_subs)
        
    dataloaders = {
        'tar': tar_loader,
        'tar_val': tar_val_loader,
        'tar_test': tar_test_loader,
        'combine_srcs': combine_srcs_loader,
        'combine_srcs_val': combine_srcs_val_loader,
        'combine_srcs_test': combine_srcs_test_loader,
        'prev_srcs': prev_srcs_loader,
        'prev_srcs_val': prev_srcs_val_loader
    }

    # load last trained model
    if test_model_name is not None:
        if is_ignore_model:
            test_model_name = prev_src_sub_model

        transfer_model, optimizer = initialize_model(args, None, args.load_source, args.n_class, pretrained_model_name = None)

        if args.load_prev_source_model:
            target_trained_model = torch.load(target_weight_path + '/' + test_model_name  + '_load.pt')
            transfer_model.load_state_dict(target_trained_model['model_state_dict'])
            optimizer.load_state_dict(target_trained_model['optimizer_state_dict'])
    
    if not args.target_evaluation_only:        
        transfer_model, each_sub_train_total_loss = train(dataloaders, transfer_model, optimizer, lamb, source_model_name, tar_model_name, target_subject, target_weight_path, timestamp, dist_measure, args, relv_sample_count)
        experiment.log_metric("Source Subjects Total Train Loss", each_sub_train_total_loss, step=count_subs)

    if args.train_model_wo_adaptation: 
        test_model_name = source_model_name 
    elif args.source_free:
        test_model_name = tar_model_name + '_source_free'
    elif args.source_combined:
        test_model_name = tar_model_name + '_source_combined' 
    elif args.train_source:
        test_model_name = source_model_name
    else:
        test_model_name = tar_model_name
    
    # transfer_model.load_state_dict(torch.load(target_weight_path + '/' + test_model_name + '.pkl'))
    transfer_model, optimizer = initialize_model(args, None, args.load_source, args.n_class, pretrained_model_name = None)
    target_trained_model = torch.load(target_weight_path + '/' + test_model_name  + '_load.pt')
    transfer_model.load_state_dict(target_trained_model['model_state_dict'])
    optimizer.load_state_dict(target_trained_model['optimizer_state_dict'])

     # *** *** Selection of relevant source samples --- ****
    if args.accumulate_prev_source_subs:
        # relv_feat_arr, relv_src_data_arr, relv_src_label_arr = create_relv_src_clusters(dataloaders['combine_srcs'], dataloaders['tar'], transfer_model, relv_feat_arr, relv_src_data_arr, relv_src_label_arr, relv_sample_count, timestamp + '/' + test_model_name)
        
        prev_relv_samples_struc = create_relv_src_clus_dbscan(dataloaders['combine_srcs'], dataloaders['tar'], transfer_model, prev_relv_samples_struc, relv_sample_count, timestamp + '/' + test_model_name)

        # prev_relv_samples_struc = create_relv_src_clus_cent(dataloaders['combine_srcs'], dataloaders['tar'], transfer_model, prev_relv_samples_struc, relv_sample_count, timestamp + '/' + test_model_name)
        # relv_feat_arr, relv_src_data_arr, relv_src_label_arr = create_relv_src_clusters_old(dataloaders['combine_srcs'], dataloaders['tar'], transfer_model, relv_feat_arr, relv_src_data_arr, relv_src_label_arr, relv_sample_count, timestamp + '/' + test_model_name)
        # relv_feat_arr, relv_src_data_arr, relv_src_label_arr = create_relv_src_dic(dataloaders['combine_srcs'], dataloaders['tar'], transfer_model, relv_feat_arr, relv_src_data_arr, relv_src_label_arr, relv_sample_count)

    if args.train_source and not args.target_evaluation_only: 
        train_loader = dataloaders['combine_srcs']
        val_loader = dataloaders['combine_srcs_val']
        test_loader = dataloaders['combine_srcs_test']
    else:
        train_loader = dataloaders['tar']
        val_loader = dataloaders['tar_val']
        test_loader = dataloaders['tar_test']

    acc = test(transfer_model, train_loader, args.batch_size, False, args.is_pain_dataset)
    acc_val = test(transfer_model, val_loader, args.batch_size, False, args.is_pain_dataset)
    acc_test = test(transfer_model, test_loader, args.batch_size, False, args.is_pain_dataset)

    experiment.log_metric("Val Accuracy", acc_val, step=count_subs)
    experiment.log_metric("Test Accuracy", acc_test, step=count_subs)

    print('Source model: ', source_model_name)
    print('Target: ', target_subject)
    print('Target model: ', tar_model_name)

    print(f'Target Accuracy: {acc}')
    print(f'Target Val Accuracy: {acc_val}')
    print(f'Target Test Accuracy: {acc_test}')

    
    return transfer_model, test_model_name, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, relv_feat_arr, relv_src_data_arr, relv_src_label_arr, prev_relv_samples_struc

def initialize_model(args, source_model_name, load_source, n_class, pretrained_model_name = None):
    transfer_model = TN(n_class, transfer_loss=transfer_loss, base_net=args.back_bone).cuda()
    # load multi-source pre-trained model and adapt to target domain 

    optimizer = torch.optim.SGD([
        {'params': transfer_model.base_network.parameters()},
        {'params': transfer_model.bottleneck_layer.parameters(), 'lr': 10 * learning_rate},
        {'params': transfer_model.classifier_layer.parameters(), 'lr': 10 * learning_rate},
    ], lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    if pretrained_model_name is not None and not load_source:
        source_trained_model = torch.load(pretrained_model_name + '_load.pt')

        # change classificaiton layer of the pre-trained model to the num of classes for the defined Architecture
        source_trained_model['model_state_dict']['classifier_layer.3.weight'] = source_trained_model['model_state_dict']['classifier_layer.3.weight'][:n_class,:]
        source_trained_model['model_state_dict']['classifier_layer.3.bias'] = source_trained_model['model_state_dict']['classifier_layer.3.bias'][:n_class]
        
        transfer_model.load_state_dict(source_trained_model['model_state_dict'])
    
    # -- load trained source model and train only target
    if load_source and source_model_name is not None and not args.single_best:
        source_trained_model = torch.load(source_model_name + '_load.pt')
        transfer_model.load_state_dict(source_trained_model['model_state_dict'])
        optimizer.load_state_dict(source_trained_model['optimizer_state_dict'])
    
    return transfer_model, optimizer

def train(dataloaders, model, optimizer, lamb, source_model_name, tar_model_name, 
          target_sub_name, target_weight_path, timestamp, dist_measure, args, relv_sample_count):
    
    target_loader, combine_srcs_loader, prev_srcs_loader  = dataloaders['tar'], dataloaders['combine_srcs'], dataloaders['prev_srcs']
    target_val_loader, combine_srcs_val_loader, prev_srcs_val_loader = dataloaders['tar_val'],dataloaders['combine_srcs_val'], dataloaders['prev_srcs_val']
    len_target_loader, len_combine_srcs_loader, len_prev_srcs_loader = len(target_loader), len(combine_srcs_loader), len(prev_srcs_loader)
    
    criterion = nn.CrossEntropyLoss()

    with experiment.train():
        experiment.log_parameter("Training", config.TRAIN_ONLY_TARGET if args.load_source else config.TRAIN_SOURCE_AND_TARGET)
        experiment.log_parameter("Back_bone", args.back_bone)
        experiment.log_parameter("Source epoch", args.source_epochs)
        experiment.log_parameter("Target epoch", args.target_epochs)
        experiment.log_parameter("Learning_rate", "{:.4}".format(float(learning_rate)))
        experiment.log_parameter("Loss", 'CrossEntropyLoss')
        experiment.log_parameter("Target model name", tar_model_name)
        experiment.log_parameter("Source model name", source_model_name)
        experiment.log_parameter("TARGET SUB NAME", target_sub_name)
        experiment.log_parameter("Optimizer", optimizer.__module__)
        experiment.log_parameter("Class", args.n_class)
        experiment.log_parameter("Transfer loss", transfer_loss)
        experiment.log_parameter("ImageNet weights", True)
        experiment.log_parameter("Pre-trained Model", args.pretrained_model)
        experiment.log_parameter("Source subjects combine", args.source_combined)
        experiment.log_parameter("Subject selection Top_s", args.top_s)
        experiment.log_parameter("Time stamp", timestamp)
        experiment.log_parameter("Distance measure", dist_mapping(dist_measure))
        experiment.log_parameter("Train using N classifier", args.train_N_source_classes)
        experiment.log_parameter("Train using Distance Measure ", args.train_with_dist_measure)
        experiment.log_parameter("Firt adapt to N source subjects", args.first_adapt_N_source_subjects)
        experiment.log_parameter("Load previous source subject model ", args.load_prev_source_model)
        experiment.log_parameter("Accumulate previous source subjects ", args.accumulate_prev_source_subs)
        experiment.log_parameter("Target dataset expanded", args.expand_tar_dataset)
        experiment.log_parameter("SEED ", _get_current_seed())
        experiment.log_parameter("Experiment description", args.experiment_description)
        experiment.log_parameter("Torch_version:", torch.__version__)
        experiment.log_parameter("Cuda_version:", torch.version.cuda)
        experiment.log_parameter("Selected relevant history samples:", relv_sample_count)
        experiment.log_parameter("Early_stop:", args.early_stop)
        
        # ------------------------------------------ ------------------------------------- #
        # Training of labeled multi-source data 
        #
        # Experiments:
        # ---- 1. Source#1 with Source#2 tested with Source#1
        # ---- 2. Source#1 with Source#2 tested with Source#2
        # ------------------------------------------ ------------------------------------- #
        each_sub_train_total_loss = 0
        if args.train_source and not args.load_source and not args.train_model_wo_adaptation:
            # n_batch = min(len_source1_loader, len_source2_loader)
            # n_batch = min(len_combine_srcs_loader, len_target_loader)
            n_batch = len_combine_srcs_loader
            print('\n ------ Start Training of Source Domain ------ \n')            
            model, _ = train_multi_model(args.source_epochs, model, combine_srcs_loader, combine_srcs_loader, optimizer, criterion, lamb, n_batch , args.early_stop, source_model_name, combine_srcs_val_loader, target_weight_path, args.oracle_setting, train_source=args.train_source)
        
        # ------------------------------------------ ------------------------------------- #
        # Generating robust target labels
        # ------------------------------------------ ------------------------------------- #
        # if args.target_clustering:
        #     # cluster_centroids = create_clusters_from_source_feat(model, combine_srcs_loader)
        #     # load source cluster centroids -- also add check when to use the existing clusters, and when to generate new once 
        #     loaded_data = np.load('source_centroids_256.npz') # define path in the function call
        #     centroids = {key: loaded_data[key] for key in loaded_data} # convert string back to dictionary
        #     n_batch = len_target_loader
        #     tar_model_name = tar_model_name + '_cluster_target' 
        #     generate_robust_target_label(model, target_loader, optimizer, criterion, args.target_epochs, n_batch, args.early_stop, tar_model_name, centroids, target_val_loader)
        #     # test(model, target_val_loader, n_batch)

        # ------------------------------------------ ------------------------------------- #
        # Training of labeled multi-source with unlabeled target model
        #
        # Experiments:
        # ---- 1. Source#1 with Target
        # ---- 2. Source#2 with Target
        # ---- 3. Source#1 + Source#2 with Target
        # ------------------------------------------ ------------------------------------- #
        if args.train_w_src_sm:
            print('\n ------ Start Training of Target Domain with Source Samples ------ \n')
            # train_multi_model(args.source_epochs, model, source1_loader, source2_loader, optimizer, criterion, lamb, n_batch , args.early_stop, source_model_name, combine_srcs_val_loader, args.oracle_setting, train_source=True)

            # train_multi_src_model_2(args.source_epochs, model, srcs_loader, optimizer, criterion, lamb, n_batch , args.early_stop, source_model_name, srcs_val_loader, args.oracle_setting, train_source=True)
            n_batch = min(len_prev_srcs_loader, len_target_loader)
            model, each_sub_train_total_loss = train_multi_model_only_src_sample(args.target_epochs, model, prev_srcs_loader, target_loader, optimizer, criterion, lamb, n_batch , args.early_stop, tar_model_name, target_val_loader, target_weight_path, args.oracle_setting, train_source=False)
           
        if not args.train_source and not args.train_w_src_sm:
            if args.source_combined:
                tar_model_name = tar_model_name + '_source_combined' 
                print('\n ----------------------------------------- -----------------------------------------------\n')
                print('\n ------ Start Training of Target Domain (Combined Source) ------ \n')
                print('\n ----------------------------------------- -----------------------------------------------\n')

            else:
                print('\n ----------------------------------------- -----------------------------------------------\n')
                print('\n ------ Start Training of Target Domain ------ \n')
                print('\n ----------------------------------------- -----------------------------------------------\n')

            n_batch = min(len_combine_srcs_loader, len_target_loader)
            model, each_sub_train_total_loss = train_multi_model(args.target_epochs, model, combine_srcs_loader, prev_srcs_loader, target_loader, optimizer, criterion, lamb, n_batch, args.early_stop, tar_model_name, target_val_loader, target_weight_path, args.oracle_setting, train_source=False)
            
        return model, each_sub_train_total_loss
def train_multi_model_only_src_sample(n_epoch, model, data_loader1, data_loader2, optimizer, 
                                      criterion, lamb, n_batch , early_stop, trained_model_name, 
                                      val_loader, target_weight_path, oracle_setting, train_source):
    best_acc = 0
    stop = 0
    srcs_avg_features = []
    threshold = 0.91

    calculate_tar_pl_ce = False
    train_total_loss_all_epochs = 0

    # SupConCriterion = SupConLoss()
    current_forzen_model = copy.deepcopy(model)

    for e in range(n_epoch):
        stop += 1
        train_loss_clf, train_loss_transfer, train_loss_total = 0, 0, 0
        train_loss_clf_domain2, train_loss_transfer_domain2, train_loss_total_domain2 = 0, 0, 0

        if train_source is False and oracle_setting is False:
            if e % 20 == 0: # remove e != 0 and
                # create a copy of a previously trained model to generate target PL
                current_forzen_model = copy.deepcopy(model)

                print("\n**** Threshold Reduced From: ", threshold)
                threshold = threshold - 0.01      
                print(" To: ", threshold)

        model.train()
        
        count = 0
        total_mmd = 0
        srcs_avg_features = []
        tar_counter = 0
        
        tar_data_iter = TragetRestartableIterator(data_loader2)
        n_batch = min(len(data_loader1), len(data_loader2))

        for data_domain1, label_domain1 in tqdm(data_loader1, leave=False):
            count = count + 1  

            ### --- *** defining for conf target samples
            # domain2 = next(conf_tar_data_iter)

            ### --- *** Current Source Subject
            # data_domain1, label_domain1 = domain1
            data_domain1, label_domain1 = data_domain1.cuda(), label_domain1.cuda()

            ### --- *** Target Subject
            tar_domain = next(tar_data_iter)
            # tar_data_domain, tar_label_domain = tar_domain
            # tar_data_domain, tar_label_domain = tar_data_domain.cuda(), tar_label_domain.cuda()

            # -- Generate target PL in minibatch
            if train_source is False and oracle_setting is False:
                for i in range(0,1):
                    target_wth_gt_labels = TensorDataset(tar_domain[0], tar_domain[1])
                    tar_loader, tar_val_loader = BaseDataset.load_target_data(target_wth_gt_labels, args.batch_size, split=False)
                    conf_data_arr, _prob_arr, pl_label_arr, _gt_arr, _, _ = create_target_pl_dicts(current_forzen_model, tar_loader, threshold, args.batch_size, args.is_pain_dataset)

                    calculate_tar_pl_ce = True
                    if len(conf_data_arr) <= 0:
                        calculate_tar_pl_ce = False
                        break

            if calculate_tar_pl_ce:
                data_domain2, label_domain2 = tar_domain
                conf_data_domain2, conf_label_domain2 = torch.tensor(conf_data_arr).cuda(), torch.tensor(pl_label_arr).cuda()
            else:
                data_domain2, label_domain2 = tar_domain
            
            data_domain2, label_domain2 = data_domain2.cuda(), label_domain2.cuda()

            # for training the custom dataset (PAIN DATASETS); I have added .float() otherwise removed it when using build-in dataset
            data_domain1 = data_domain1.float()
            data_domain2 = data_domain2.float()

            # calculate target PL loss only with conf samples
            clf_loss, transfer_loss = 0, 0
            if calculate_tar_pl_ce and count <= len(tar_loader):
                label_source_pred, transfer_loss, domain1_feature = model(data_domain1, conf_data_domain2)
                clf_loss = criterion(label_source_pred, label_domain1)
            else:
                label_source_pred, transfer_loss, _ = model(data_domain1, data_domain2)
                clf_loss = criterion(label_source_pred, label_domain1)

            # loss = (clf_loss) + lamb * transfer_loss
            transfer_loss = transfer_loss.detach().item() if transfer_loss and transfer_loss.detach().item() == transfer_loss.detach().item() else 0 # to avoid 'NaN'
            loss = (clf_loss) + transfer_loss

            
            # loss = (clf_loss) 

            total_mmd += transfer_loss 
            # loss.backward()

            # adding target loss with source loss
            if train_source or oracle_setting:
                label_pred_domain2, transfer_loss_domain2, domain2_feature  = model(data_domain2, data_domain1)
                clf_loss_domain2 = criterion(label_pred_domain2, label_domain2)
                loss_domain2 = (clf_loss_domain2) + lamb * transfer_loss_domain2

                """
                    combine source-1 and source-2 loss
                """
                # combine_loss = loss_domain2 + loss
                combine_loss = clf_loss_domain2 + loss
                optimizer.zero_grad()
                combine_loss.backward()

                if domain1_feature.shape == domain2_feature.shape:
                    srcs_avg_features.append(torch.mean(torch.stack([domain1_feature, domain2_feature]), dim=0))

            else:
                if calculate_tar_pl_ce and count <= len(tar_loader) :
                    label_pred_domain2, transfer_loss_domain2, domain2_feature  = model(conf_data_domain2, None)
                    clf_loss_domain2 = criterion(label_pred_domain2, conf_label_domain2)
                    loss = clf_loss_domain2 + loss
                    
                    optimizer.zero_grad()
                    loss.backward()

                    tar_counter = tar_counter + 1
                    train_loss_clf_domain2 = clf_loss_domain2.detach().item() + train_loss_clf_domain2
                else:
                    train_loss_clf_domain2 = 0
                    optimizer.zero_grad()
                    loss.backward()
                
            optimizer.step()
            train_loss_clf = clf_loss.detach().item() if clf_loss else 0 + train_loss_clf + train_loss_clf_domain2
            train_loss_transfer = transfer_loss  + train_loss_transfer
            train_loss_total = train_loss_clf + train_loss_transfer

            if train_source:
                train_loss_clf_domain2 = clf_loss_domain2.detach().item() + train_loss_clf_domain2
                train_loss_transfer_domain2 = transfer_loss_domain2.detach().item() + train_loss_transfer_domain2
                train_loss_total_domain2 = loss_domain2.detach().item() + train_loss_total_domain2

        acc = test(model, val_loader, args.batch_size, False, args.is_pain_dataset)
        
        if train_loss_clf_domain2 > 0:
            print(f'Epoch: [{e:2d}/{n_epoch}], train_src_loss_clf: {train_loss_clf/n_batch:.4f}, train_tar_pl_loss_clf: {train_loss_clf_domain2/len(tar_loader):.4f}, transfer_loss: {train_loss_transfer/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}')
        else:
            print(f'Epoch: [{e:2d}/{n_epoch}], train_src_loss_clf: {train_loss_clf/n_batch:.4f}, transfer_loss: {train_loss_transfer/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}')

        experiment.log_metric("Source-1 loss:", train_loss_clf/n_batch, epoch=e)
        experiment.log_metric("Total loss:", train_loss_total/n_batch, epoch=e)
        if train_loss_clf_domain2 > 0 and len(tar_loader) > 0:
            experiment.log_metric("Target pl train loss:", train_loss_clf_domain2/len(tar_loader))

        experiment.log_metric("Each epoch top 1 accuracy", acc, epoch=e)

        # add up all the epochs total losses
        train_total_loss_all_epochs = train_total_loss_all_epochs + train_loss_total/n_batch

        if best_acc < acc:
            best_acc = acc
            trained_model_name = trained_model_name + ''
            torch.save(model.state_dict(), target_weight_path + '/' + trained_model_name + '.pkl')
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, target_weight_path + '/' + trained_model_name + '_load.pt')
            # save source feature maps
            if len(srcs_avg_features) > 0:
                torch.save(srcs_avg_features, target_weight_path + '/' + trained_model_name + '_features.pt')
            experiment.log_metric("Val Target Best Accuracy", best_acc, epoch=e)
            stop = 0
        if stop >= early_stop:
            break
    print(total_mmd)

    print("Best Val Acc: ", best_acc)
    train_total_loss_all_epochs = train_total_loss_all_epochs/n_epoch
    return model, train_total_loss_all_epochs
        
def train_multi_model(n_epoch, model, data_loader1, prev_src_loader, data_loader2, 
                      optimizer, criterion, lamb, n_batch , early_stop, trained_model_name, 
                      val_loader, target_weight_path, oracle_setting, train_source):
    best_acc = 0
    stop = 0
    srcs_avg_features = []
    threshold = 0.91

    calculate_tar_pl_ce = False
    train_total_loss_all_epochs = 0

    # SupConCriterion = SupConLoss()
    current_forzen_model = copy.deepcopy(model)

    # train_source = False # remove this line ; ITS ONLY THERE TO PERFORM EXPERIMENTS ON THE MMD LOSS FOR DOMAIN SHIFT FOR GDA FOR DGA-1033 PRESENTATION
    for e in range(n_epoch):
        stop += 1
        train_loss_clf, train_loss_transfer, train_loss_total = 0, 0, 0
        train_loss_clf_domain2, train_loss_transfer_domain2, train_loss_total_domain2 = 0, 0, 0
        # calculate_tar_pl_ce = False

        if train_source is False and oracle_setting is False:
            if e % 20 == 0: # remove e != 0 and
                # create a copy of a previously trained model to generate target PL
                current_forzen_model = copy.deepcopy(model)

                print("\n**** Threshold Reduced From: ", threshold)
                threshold = threshold - 0.01      
                print(" To: ", threshold)

        model.train()
        
        count = 0
        total_mmd = 0
        srcs_avg_features = []
        tar_counter = 0
        
        tar_data_iter = TragetRestartableIterator(data_loader2)
        n_batch = min(len(data_loader1), len(prev_src_loader))

        for (src_domain, prev_src_domain) in zip(data_loader1, prev_src_loader):

            count = count + 1  

            ### --- *** Current Source Subject
            data_domain1, label_domain1 = src_domain
            data_domain1, label_domain1 = data_domain1.cuda(), label_domain1.cuda()

            ### --- *** Previous Source Subjects
            prev_data_domain, prev_label_domain = prev_src_domain
            prev_data_domain, prev_label_domain = prev_data_domain.cuda(), prev_label_domain.cuda()

            ### --- *** Target Subject
            tar_domain = next(tar_data_iter)

            # -- Generate target PL in minibatch
            if train_source is False and oracle_setting is False:
                for i in range(0,1):
                    target_wth_gt_labels = TensorDataset(tar_domain[0], tar_domain[1])
                    tar_loader, tar_val_loader = BaseDataset.load_target_data(target_wth_gt_labels, args.batch_size, split=False)
                    conf_data_arr, _prob_arr, pl_label_arr, _gt_arr, _, _ = create_target_pl_dicts(current_forzen_model, tar_loader, threshold, args.batch_size, args.is_pain_dataset)

                    calculate_tar_pl_ce = True
                    if len(conf_data_arr) <= 0:
                        calculate_tar_pl_ce = False
                        break

             # defining for domain-2
            if train_source or oracle_setting:
                data_domain2, label_domain2 = tar_domain
                data_domain2, label_domain2 = data_domain2.cuda(), label_domain2.cuda()
            else:
                if calculate_tar_pl_ce:
                    data_domain2, label_domain2 = tar_domain
                    conf_data_domain2, conf_label_domain2 = torch.tensor(conf_data_arr).cuda(), torch.tensor(pl_label_arr).cuda()
                else:
                    data_domain2, label_domain2 = tar_domain
                
                data_domain2, label_domain2 = data_domain2.cuda(), label_domain2.cuda()

            # for training the custom dataset (PAIN DATASETS); I have added .float() otherwise removed it when using build-in dataset
            data_domain1 = data_domain1.float()
            data_domain2 = data_domain2.float()
            prev_data_domain = prev_data_domain.float()
            prev_label_domain = prev_label_domain.to(torch.int64)

            # calculate target PL loss only with conf samples
            clf_loss, prev_clf_loss, transfer_loss = 0, 0, 0
            if calculate_tar_pl_ce and count <= len(tar_loader) :
                # calculate mmd loss with prev source sub
                prev_label_source_pred, prev_transfer_loss, _ = model(prev_data_domain, data_domain1)
                prev_clf_loss = criterion(prev_label_source_pred, prev_label_domain)

                label_source_pred, transfer_loss, domain1_feature = model(data_domain1, conf_data_domain2)
                clf_loss = criterion(label_source_pred, label_domain1)
                transfer_loss = transfer_loss.detach().item() if transfer_loss and transfer_loss.detach().item() == transfer_loss.detach().item() else 0 # to avoid 'NaN'
            else:
                label_source_pred, prev_transfer_loss, _ = model(data_domain1, prev_data_domain)
                clf_loss = criterion(label_source_pred, label_domain1)
               
            prev_transfer_loss = prev_transfer_loss.detach().item() if prev_transfer_loss and prev_transfer_loss.detach().item() == prev_transfer_loss.detach().item() else 0 # to avoid 'NaN'
            loss = (clf_loss) + prev_clf_loss + transfer_loss + (lamb*prev_transfer_loss)

            total_mmd += transfer_loss 

            # adding target loss with source loss
            if train_source or oracle_setting:
                label_pred_domain2, transfer_loss_domain2, domain2_feature  = model(data_domain2, data_domain1)
                clf_loss_domain2 = criterion(label_pred_domain2, label_domain2)
                loss_domain2 = (clf_loss_domain2) + lamb * transfer_loss_domain2

                """
                    combine source-1 and source-2 loss
                """
                # combine_loss = loss_domain2 + loss
                combine_loss = clf_loss_domain2 + loss
                optimizer.zero_grad()
                combine_loss.backward()

                if domain1_feature.shape == domain2_feature.shape:
                    srcs_avg_features.append(torch.mean(torch.stack([domain1_feature, domain2_feature]), dim=0))

            else:
                if calculate_tar_pl_ce and count <= len(tar_loader) :
                    label_pred_domain2, transfer_loss_domain2, domain2_feature  = model(conf_data_domain2, None)
                    clf_loss_domain2 = criterion(label_pred_domain2, conf_label_domain2)
                    loss = clf_loss_domain2 + loss
                    
                    optimizer.zero_grad()
                    loss.backward()

                    tar_counter = tar_counter + 1
                    train_loss_clf_domain2 = clf_loss_domain2.detach().item() + train_loss_clf_domain2
                else:
                    optimizer.zero_grad()
                    loss.backward()
                
            optimizer.step()
            train_loss_clf = clf_loss.detach().item() if clf_loss else 0 + prev_clf_loss.detach().item() if prev_clf_loss else 0 + train_loss_clf
            train_loss_transfer = transfer_loss + prev_transfer_loss + train_loss_transfer
            train_loss_total = train_loss_clf + train_loss_transfer

            # target loss_clf
            if train_source:
                train_loss_clf_domain2 = clf_loss_domain2.detach().item() + train_loss_clf_domain2
                train_loss_transfer_domain2 = transfer_loss_domain2.detach().item() + train_loss_transfer_domain2
                train_loss_total_domain2 = loss_domain2.detach().item() + train_loss_total_domain2

        acc = test(model, val_loader, args.batch_size, False, args.is_pain_dataset)
        
        if train_loss_clf_domain2 > 0:
            print(f'Epoch: [{e:2d}/{n_epoch}], train_src_loss_clf: {train_loss_clf/n_batch:.4f}, train_tar_pl_loss_clf: {train_loss_clf_domain2/len(tar_loader):.4f}, transfer_loss: {train_loss_transfer/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}')
        else:
            print(f'Epoch: [{e:2d}/{n_epoch}], train_src_loss_clf: {train_loss_clf/n_batch:.4f}, transfer_loss: {train_loss_transfer/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}')

        experiment.log_metric("Source-1 loss:", train_loss_clf/n_batch, epoch=e)
        experiment.log_metric("Total loss:", train_loss_total/n_batch, epoch=e)
        if train_loss_clf_domain2 > 0 and len(tar_loader) > 0:
            experiment.log_metric("Target pl train loss:", train_loss_clf_domain2/len(tar_loader))

        experiment.log_metric("Each epoch top 1 accuracy", acc, epoch=e)

        # add up all the epochs total losses
        train_total_loss_all_epochs = train_total_loss_all_epochs + train_loss_total/n_batch

        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), target_weight_path + '/' + trained_model_name + '.pkl')
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, target_weight_path + '/' + trained_model_name + '_load.pt')
            # save source feature maps
            if len(srcs_avg_features) > 0:
                torch.save(srcs_avg_features, target_weight_path + '/' + trained_model_name + '_features.pt')
            experiment.log_metric("Val Target Best Accuracy", best_acc, epoch=e)
            stop = 0
        if stop >= early_stop:
            break
    print(total_mmd)

    # visualize_tsne(clusters_by_label)
    print("Best Val Acc: ", best_acc)
    train_total_loss_all_epochs = train_total_loss_all_epochs/n_epoch
    return model, train_total_loss_all_epochs

def measure_src_samples_tar_dist(model, srcs_sub_loader, tar_sub_loader, dist_measure, tar_name, batch_size):
    model.eval()

    mmd = MMD_loss()
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    total_dist = 0
    tar_mean = 0.0
    concat_features = []
    closest_src_samples = []
    closest_src_labels = []
    closest_src_dists = []
    relv_sample_count = 50000
    with torch.no_grad():
        tar_data_iter = TragetRestartableIterator(tar_sub_loader)
        n_batch = len(srcs_sub_loader)

        for src_data, src_gt in tqdm(srcs_sub_loader, leave=False):
            # count = count + 1  

            target = next(tar_data_iter)

            # src_data, src_gt = sources
            src_data, src_gt = src_data.cuda(), src_gt.cuda()
            src_data = src_data.float()
            src_feat = model.forward_features(src_data)

            tar_data, _ = target
            tar_data = tar_data.cuda()
            tar_data = tar_data.float()
            tar_feat = model.forward_features(tar_data)

            # concat_features = torch.cat([concat_features, tar_feat], dim=0) if len(concat_features) > 0 else tar_feat 

            if dist_measure == config.MMD_SIMILARITY:
                mmd_dist = mmd(src_feat, tar_feat)
                total_dist = total_dist + mmd_dist.cpu().numpy()
            elif dist_measure == config.COSINE_SIMILARITY:
                cosine_dist = cosine_sim(src_feat, tar_feat)
                # sim = cosine_sim(src_feat, tar_feat[0].unsqueeze(0).expand(src_feat.shape[0], -1))

            closest_src_samples.extend(src_data.detach().cpu().numpy())
            closest_src_labels.extend(src_gt.detach().cpu().numpy())
            closest_src_dists.extend(cosine_dist.detach().cpu().numpy())


    np.savez(tar_name, closest_src_samples=closest_src_samples, closest_src_labels=closest_src_labels, closest_src_dists=closest_src_dists)

    # batches = min(len(srcs_sub_loader), len(tar_sub_loader))
    # total_dist = total_dist / batches
    # print(total_dist)
    # return closest_src_samples, closest_src_labels

    # feat_memory_bank = np.concatenate((feat_memory_bank, np.column_stack((features.cpu().detach().numpy(), target.cpu().detach().numpy()))), axis=0) if len(feat_memory_bank) > 0 else np.column_stack((features.cpu().detach().numpy(), target.cpu().detach().numpy()))

def measure_srcs_tar_dist(model, srcs_sub_loader, tar_sub_loader, dist_measure, batch_size):
    model.eval()

    mmd = MMD_loss()
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    total_dist = 0
    tar_mean = 0.0
    concat_features = []
    with torch.no_grad():
        for (sources, target) in zip(srcs_sub_loader, tar_sub_loader):
            src_data, src_gt = sources
            src_data, src_gt = src_data.cuda(), src_gt.cuda()
            src_data = src_data.float()
            src_feat = model.forward_features(src_data)

            tar_data, _ = target
            tar_data = tar_data.cuda()
            tar_data = tar_data.float()
            tar_feat = model.forward_features(tar_data)

            concat_features = torch.cat([concat_features, tar_feat], dim=0) if len(concat_features) > 0 else tar_feat 

            if dist_measure == config.MMD_SIMILARITY:
                mmd_dist = mmd(src_feat, tar_feat)
                total_dist = total_dist + mmd_dist.cpu().numpy()
            elif dist_measure == config.COSINE_SIMILARITY:
                cosine_dist = cosine_sim(src_feat, tar_feat)
                total_dist = total_dist + (np.sum(cosine_dist.cpu().detach().numpy()))/batch_size
            else:
                tensor1_probs = F.softmax(src_feat, dim=1)
                tensor2_probs = F.softmax(tar_feat, dim=1)
                total_dist = total_dist + F.kl_div(tensor1_probs.log(), tensor2_probs, reduction='batchmean')

    # print(cosine_dist)
    batches = min(len(srcs_sub_loader), len(tar_sub_loader))
    total_dist = total_dist / batches
    print(total_dist)
    return total_dist

def test(model, target_test_loader, batch_size, top_N_tar_evaluate=False, is_pain_dataset=False):
    model.eval()
    correct = 0
    corr_acc_top1 = 0
    len_target_dataset = len(target_test_loader) if is_pain_dataset else len(target_test_loader.dataset) 
    acc_top1 = 0
    store_pred = []

    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.cuda(), target.cuda()
            s_output = model.predict(data.float())
            pred = torch.max(s_output, 1)[1]

            if top_N_tar_evaluate:
                store_pred.extend(pred.tolist())
            else:
                correct += torch.sum(pred == target)
                
            if not top_N_tar_evaluate:
                acc_top1 = Accuracy(task='multiclass', num_classes=args.n_class, top_k=1).to(device)
                corr_acc_top1 += acc_top1(s_output, target)
    
    if top_N_tar_evaluate:
        item_counts = Counter(store_pred)
        print(item_counts)
        return item_counts
    
    acc = correct.double() / len_target_dataset
    # batch_samples = len_target_dataset / int(batch_size)
    batch_samples = len_target_dataset if len_target_dataset == len(target_test_loader) else len_target_dataset / int(batch_size)
    
    acc_top1 = corr_acc_top1/batch_samples
        
    return acc_top1

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on FER')
    arg_parser.add_argument('--src_train_datasets_path', type=str, default=config.BIOVID_SUBS_PATH)
    arg_parser.add_argument('--src_test_datasets_path', type=str, default=config.BIOVID_SUBS_PATH)

    arg_parser.add_argument('--pain_db_root_path', type=str, default=config.BIOVID_PATH)

    arg_parser.add_argument('--pretrained_model', type=str, default='mcmaster_trained_model')
    arg_parser.add_argument('--batch_size', type=int, default=16)    # 1 only because JAFFE has small dataset
    arg_parser.add_argument('--source_epochs', type=int, default=20)
    arg_parser.add_argument('--target_epochs', type=int, default=30)
    arg_parser.add_argument('--early_stop', type=int, default=20)
    arg_parser.add_argument('--single_best', type=str, default=False)
    arg_parser.add_argument('--train_model_wo_adaptation', type=str, default=False)
    arg_parser.add_argument('--oracle_setting', type=str, default=False)
    arg_parser.add_argument('--source_free', type=str, default=False)
    arg_parser.add_argument('--target_clustering', type=bool, default=False)

    arg_parser.add_argument('--load_source', type=str, default=False)
    arg_parser.add_argument('--train_N_source_classes', type=bool, default=False)
    arg_parser.add_argument('--train_with_dist_measure', type=bool, default=True)
    arg_parser.add_argument('--first_adapt_N_source_subjects', type=bool, default=True)
    arg_parser.add_argument('--load_prev_source_model', type=bool, default=True)
    arg_parser.add_argument('--accumulate_prev_source_subs', type=bool, default=True)

    arg_parser.add_argument('--tar_subject', type=int, help='Target subject number', default=4)
    arg_parser.add_argument('--expand_tar_dataset', type=bool, default=False)

    arg_parser.add_argument('--experiment_description', type=str, default='2loaders CS threshold=0.9. create src & target clusters dbscan. *Add tar PL in minibatch. *MMD Conf tar samples.**') # create src and target clusters k-means *Add prev src CE 

    arg_parser.add_argument('--target_evaluation_only', type=bool, default=False)
    arg_parser.add_argument('--top_src_sub', type=str, default='072414_m_23,082714_m_22,110909_m_29')
    arg_parser.add_argument('--top_timestamp', type=str, default='1721256232')
    arg_parser.add_argument('--train_source', type=str, default=False)
    arg_parser.add_argument('--weights_folder', type=str, default=config.WEIGHTS_FOLDER)
    arg_parser.add_argument('--source_combined', type=str, default=False)
    arg_parser.add_argument('--is_pain_dataset', type=bool, default=True)
    arg_parser.add_argument('--dist_measure', type=str, default=config.COSINE_SIMILARITY)
    arg_parser.add_argument('--top_s', type=int, default=11)
    arg_parser.add_argument('--n_class', type=int, default=2) # 2 for pain Biovid -- n_class = 77: to train a classifier with N source subject classes
    arg_parser.add_argument('--n_class_N_src', type=int, default=77)
    arg_parser.add_argument('--top_rev_src_sam', type=int, default=31000)
    arg_parser.add_argument('--train_w_rand_src_sm', type=bool, default=False)
    arg_parser.add_argument('--train_w_src_sm', type=bool, default=False)
    arg_parser.add_argument('--cs_threshold', type=float, default=0.90)
    
    arg_parser.add_argument('--back_bone', default="resnet18", type=str)
    args = arg_parser.parse_args()

    #-- BioVid
    # 0. 081014_w_27 [40]
    # 1. 101609_m_36 [70]
    # 2. 112009_w_43 [66]
    # 3. 091809_w_43 [68]
    # 4. 071309_w_21 [4]
    # 5. 073114_m_25 [69]
    # 6. 080314_w_25 [80]
    # 7. 073109_w_28 [82]
    # 8. 100909_w_65 [13]
    # 9. 081609_w_40 [17]
    
    #-- UNBC
    # 0. 107-hs107 
    # 1. 109-ib109
    # 2. 121-vw121
    # 3. 123-jh123
    # 4. 115-jy115
    # print("\n\n*** Remove Detached =============\n\n ")
    main(args)