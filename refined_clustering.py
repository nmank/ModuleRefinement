import center_algorithms as ca

import utils as utl

import orthrus
from orthrus import core
from orthrus.core import dataset, helper

from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd

import os

'''
TO DO:

'''

def load_data(project: str) -> tuple:

    #read dataset
    ds = dataset.load_dataset('/data4/zoetis/Data/TPM_C1_Z34_Z40_Z42_Z75.ds')
    sample_ids  = ds.metadata['Project'] == project

    the_dataset = ds.slice_dataset(sample_ids=sample_ids)

    data = the_dataset.data

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    labels = the_dataset.metadata['Diagnosis']
    skf.get_n_splits(data, labels)
    
    return data, labels, skf

def load_modules(file_path: str) -> tuple:
    the_modules = helper.load_object(file_path)
    all_features = set()
    for _, row in the_modules.iterrows():
        all_features = set(row.item()).union(all_features)

    return the_modules, all_features

def process_data(split_module_data: pd.DataFrame, center_method: str) -> list:

    X = np.array(split_module_data)

    normalized_data = []

    if center_method == 'flag_median' or center_method == 'flag_mean':
        #make unit vectors
        for row in X.T:
            row_norm = np.linalg.norm(row)
            if row_norm != 0:
                normalized_data.append(np.expand_dims(row / row_norm, axis = 1))
            else:
                print('zero column in module data. this column was removed.')
    elif center_method == 'eigengene':
        #mean center the columns
        p = X.shape[0]
        column_means = np.repeat(np.expand_dims(np.mean(X, axis = 0), axis = 1).T, p, axis = 0)
        X = X - column_means
        normalized_data = [np.expand_dims(d, axis = 1) for d in X.T]
    else:
        #no normalization
        normalized_data = [np.expand_dims(d, axis = 1) for d in X.T]


    return normalized_data

def initialize_centers(the_modules: pd.DataFrame, split_module_data: pd.DataFrame, center_method: str) -> list:
    dimension = int(center_method[-1])

    initial_centers = []
    for _, module in the_modules.iterrows():

        module_features = module.item()
        module_data = split_module_data[module_features]

        #change this depending on method
        module_data = [np.expand_dims(m, axis = 1) for m in np.array(module_data).T]
        

        if center_method[:-1] == 'eigengene':
            center = ca.eigengene(module_data, dimension)
        else:
            module_data = [d/np.linalg.norm(d) for d in module_data]

            opt_dim = ca.find_optimal_dimension(module_data)
            print(opt_dim)
            if center_method[:-1] == 'flag_mean':
                center = ca.flag_mean(module_data, dimension)
            elif center_method[:-1] == 'flag_median':
                center = ca.irls_flag(module_data, dimension, 50, 'sine')[0]

        initial_centers.append(center)
    return initial_centers

def save_modules(normalized_data: list, split_module_data: pd.DataFrame, centers: list, save_path: str) -> None:
    the_modules = pd.DataFrame(columns = ['Feature Set'])
    for module_number in range(len(centers)):
        d_mat = ca.distance_matrix(normalized_data, centers, True)
        index  = np.argmax(d_mat, axis = 0)

        genes_in_one_module = list(split_module_data.T[index == module_number].index)
        row = pd.DataFrame(columns = ['Feature Set'], data = [[genes_in_one_module]])
        the_modules = pd.concat([the_modules, row])
        # the_modules = the_modules.append(row, ignore_index = True)

            
        
        helper.save_object(the_modules, save_path, overwrite=True)

def run_lbg_clustering(normalized_data: list, initial_centers: list, center_method: str, labels: list = None) -> list:
    dimension = int(center_method[-1])

    the_opt_type = center_method[:-1]

    if the_opt_type == 'flag_median':
        the_opt_type = 'sine'
    elif the_opt_type == 'flag_mean':
        the_opt_type = 'sinesq'
    
    if the_opt_type != 'zobs_eigengene':
        labels = None

    
    outstuff = ca.lbg_subspace(normalized_data, epsilon=.0001, centers = initial_centers, 
                              n_centers = len(initial_centers), opt_type = the_opt_type, 
                              n_its = 10, seed = 1, r = dimension, similarity = True, labels = labels)

    centers = outstuff[0]
    return centers

def refined_modules(split_data: pd.DataFrame, module_path: str, center_methods: list = []):
    split_path = module_path.split('/')
    save_prefix = 

    the_modules = load_modules(module_path)
    for center_method in center_methods:
        save_path0 = f'./refined_modules/{center_method}'
        if not os.isdir(save_path0):
            os.mkdir(save_path0)
        
        save_path1 = f'{save_path0}/{split_path[2]}'
        if not os.isdir(save_path1):
            os.mkdir(save_path1)

        save_path =  f'{save_path1}/{split_path[3][:-4]}.csv'

        normalized_split_data = process_data(split_data, center_method)
        initial_centers = initialize_centers(the_modules, normalized_split_data, center_method)
        final_centers = run_lbg_clustering(normalized_split_data, initial_centers, center_method)
        save_modules(normalized_split_data, split_data, final_centers, save_path)
        
    print('foo')

if __name__ == '__main__':

    center_methods = ['eigengene1', 
                      'flag_mean1', 
                      'flag_median1', 
                      'eigengene4',
                      'flag_mean4',
                      'flag_median4']

    prms = {}

    data_dir = './data/'
    for file_name in os.listdir(data_dir):

        if 'label' not in file_name:

            print('------------------------')
            print(f'computing {file_name[:-4]} modules')

            if 'gse' in file_name:
                species = 'human'
            else:
                species = 'mouse'

            class_data, unique_labels, data_all, labels_all = utl.load_data(data_dir +file_name)

            module_file = f'./modules/all/{file_name}'
            prms[file_name[:-4]] = refined_modules(data_all, module_file, center_methods)

            for dta, lbl in zip(class_data, unique_labels):
                module_file = f'./modules/all/{file_name[:-4]}_{lbl}.csv'
                prms[file_name[:-4]] = refined_modules(dta, module_file, center_methods)

            print(f'computing 5 fold modules...')

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            skf.get_n_splits(data_all, labels_all)

            fold_number = 0
            for train_index, test_index in skf.split(data_all, labels_all):
                split_data = data_all.iloc[train_index]
                split_labels = labels_all.iloc[train_index]

                module_file = f'./modules/5fold/fold{fold_number}_{file_name}'
                prms[file_name[:-4]] = refined_modules(split_data, module_file, center_methods)

                class_data, unique_labels = utl.separate_classes(split_data, split_labels)

                for dta, lbl in zip(class_data, unique_labels):
                    module_file = f'./modules/5fold/fold{fold_number}_{file_name[:-4]}_{lbl}.csv'
                    prms[file_name[:-4]] = refined_modules(dta, module_file, center_methods)

                fold_number += 1

    wgcna_rsquared = pd.DataFrame.from_dict(prms, orient="index")
    wgcna_rsquared.to_csv("wgcna_rquared.csv")

