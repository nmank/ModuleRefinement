import center_algorithms as ca

import utils as utl

import orthrus
from orthrus import core
from orthrus.core import dataset, helper

from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd

import os

import ModuleLBG as mlbg

'''
TO DO:

'''


def refined_modules(split_data: pd.DataFrame, module_path: str, center_methods: list = []):
    split_path = module_path.split('/')

    the_modules, _ = utl.load_modules(module_path)
    for center_method in center_methods:
        save_path0 = f'./refined_modules/{center_method}'
        if not os.path.isdir(save_path0):
            os.mkdir(save_path0)
        
        save_path1 = f'{save_path0}/{split_path[2]}'
        if not os.path.isdir(save_path1):
            os.mkdir(save_path1)

        save_path =  f'{save_path1}/{split_path[3][:-4]}.pickle'

        split_data = split_data.loc[:, (split_data != 0).any(axis=0)] #remove columns with all 0s

        #make index here! this is who's in what module
        feature_names = list(split_data.columns)
        feature_labels = pd.DataFrame(columns = feature_names, data = np.zeros((1,len(feature_names))))
        ii=0
        for _, m in the_modules.iterrows():
            feature_labels[m] = ii
            ii+=1
        index = feature_labels.iloc[0]

        restricted_data = split_data[feature_names]

        my_mlbg = mlbg.ModuleLBG(center_method = center_method[:-1], dimension = center_method[-1],
                distance = 'correlation', centrality = 'degree')

        normalized_split_data = my_mlbg.process_data(np.array(restricted_data))

        initial_centers = my_mlbg.calc_centers(normalized_split_data, index)
        my_mlbg.fit_transform(normalized_split_data)

        labels = my_mlbg.get_labels(normalized_split_data)

        utl.save_modules(restricted_data, labels, save_path)
        

if __name__ == '__main__':

    center_methods = ['eigengene1', 
                      'flag_mean1', 
                      'flag_median1', 
                      'module_expression1']

    prms = {}

    data_dir = './data/'
    for file_name in os.listdir(data_dir):

        if 'label' not in file_name and 'gse' not in file_name:

            print('------------------------')
            print(f'computing {file_name[:-4]} modules')

            if 'gse' in file_name:
                species = 'human'
            else:
                species = 'mouse'

            class_data, unique_labels, data_all, labels_all = utl.load_data(data_dir +file_name)

            module_file = f'./modules/all/{file_name}'
            refined_modules(data_all, module_file, center_methods)

            for dta, lbl in zip(class_data, unique_labels):
                module_file = f'./modules/all/{file_name[:-4]}_{lbl}.csv'
                refined_modules(dta, module_file, center_methods)

            print(f'computing 5 fold modules...')

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            skf.get_n_splits(data_all, labels_all)

            fold_number = 0
            for train_index, test_index in skf.split(data_all, labels_all):
                split_data = data_all.iloc[train_index]
                split_labels = labels_all.iloc[train_index]

                module_file = f'./modules/5fold/fold{fold_number}_{file_name}'
                refined_modules(split_data, module_file, center_methods)

                class_data, unique_labels = utl.separate_classes(split_data, split_labels)

                for dta, lbl in zip(class_data, unique_labels):
                    module_file = f'./modules/5fold/fold{fold_number}_{file_name[:-4]}_{lbl}.csv'
                    refined_modules(dta, module_file, center_methods)

                fold_number += 1

