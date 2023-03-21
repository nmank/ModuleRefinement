

import orthrus
from orthrus import core
from orthrus.core import dataset, helper

from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd

import os

import argparse

import sys

import ModuleRefinement.utils as utl



'''
Computing refined modules.
'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default = 'data', type = str, help = "path to data directory")
    parser.add_argument("--modules_dir", default = 'experiments/modules', type = str, help = "path to modules directory")
    parser.add_argument("--refined_modules_dir", default = 'experiments/refined_modules_sanity', type = str, help = "path to refined modules directory")
    parser.add_argument("--center_methods_file", default = 'experiments/compute_modules/center_methods_sanity.csv', type = str, 
                        help = "path to csv file with 3 columns: 1) center type, 2) center dimension, 3) data dimension")
    args = parser.parse_args()

    data_dir = args.data_dir
    modules_dir = args.modules_dir
    refined_modules_dir = args.refined_modules_dir
    center_methods_file = args.center_methods_file

    center_methods = np.array(pd.read_csv(center_methods_file, index_col = 0))

    prms = {}

    for file_name in os.listdir(data_dir):

        if 'label' not in file_name:

            print('------------------------')
            print(f'computing {file_name[:-4]} modules')

            if 'gse' in file_name:
                species = 'human'
            else:
                species = 'mouse'

            class_data, unique_labels, data_all, labels_all = utl.load_data(f'{data_dir}/{file_name}')

            module_file = f'{modules_dir}/all/{file_name[:-4]}.pickle'
            utl.refined_modules(data_all, module_file, center_methods, refined_modules_dir)

            for dta, lbl in zip(class_data, unique_labels):
                module_file = f'{modules_dir}/all/{file_name[:-4]}_{lbl}.pickle'
                utl.refined_modules(dta, module_file, center_methods, refined_modules_dir)

            print(f'computing 5 fold modules...')

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            skf.get_n_splits(data_all, labels_all)

            fold_number = 0
            for train_index, test_index in skf.split(data_all, labels_all):
                split_data = data_all.iloc[train_index]
                split_labels = labels_all.iloc[train_index]

                module_file = f'{modules_dir}/5fold/fold{fold_number}_{file_name[:-4]}.pickle'
                utl.refined_modules(split_data, module_file, center_methods, refined_modules_dir)

                class_data, unique_labels = utl.separate_classes(split_data, split_labels)

                for dta, lbl in zip(class_data, unique_labels):
                    module_file = f'{modules_dir}/5fold/fold{fold_number}_{file_name[:-4]}_{lbl}.pickle'
                    utl.refined_modules(dta, module_file, center_methods, refined_modules_dir)

                fold_number += 1

