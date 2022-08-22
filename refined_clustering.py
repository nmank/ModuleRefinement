import sys

sys.path.append('/home/katrina/a/mankovic/FlagIRLS')

import center_algorithms as ca

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

def initialize_centers(the_modules: pd.DataFrame, split_module_data: pd.DataFrame, center_method: str, labels: np.array = None) -> list:
    dimension = int(center_method[-1])

    initial_centers = []
    for _, module in the_modules.iterrows():

        module_features = module.item()
        module_data = split_module_data[module_features]

        #change this depending on method
        module_data = [np.expand_dims(m, axis = 1) for m in np.array(module_data).T]
        

        if center_method[:-1] == 'eigengene':
            center = ca.eigengene(module_data, dimension)
        elif center_method[:-1] == 'zobs_eigengene':
            center = ca.zobs_eigengene(module_data, dimension, labels)
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


if __name__ == '__main__':
    project_name = 'Z75'
    do_AD = True
    modules_directory = '/data4/zoetis/shared/mank_experiments/figures'
    figures_or_dump = 'dump'

    # center_methods = ['eigengene1', 
    #                   'flag_mean1', 
    #                   'flag_median1', 
    #                   'eigengene4',
    #                   'flag_mean4',
    #                   'flag_median4']
    center_methods = ['eigengene8',
                      'flag_mean8',
                      'flag_median8']
    # center_methods = ['zobs_eigengene1']
    # center_methods = ['flag_median0']


    #build directories
    for center_method in center_methods:
        if do_AD:
            out_directory =f'/data4/zoetis/shared/mank_experiments/{figures_or_dump}/{project_name}/AD_{center_method}_modules'
        else:
            out_directory =f'/data4/zoetis/shared/mank_experiments/{figures_or_dump}/{project_name}/{center_method}_modules'

        try:
            os.mkdir(out_directory)
        except FileExistsError:
            print('Directory Exists')


    data, labels, skf = load_data(project = project_name)

    fold_number = 0
    for train_index, test_index in skf.split(data, labels):

        split_data = data.iloc[train_index]  

        if do_AD == True:
            split_data = split_data[labels == 'AD']

        split_labels = np.array([int(l == 'AD') for l in labels[train_index]])

        if do_AD:
            module_file_path = f'{modules_directory}/{project_name}/AD_wgcna_modules/modules_fold{fold_number}.pickle'
        else:
            module_file_path = f'{modules_directory}/{project_name}/wgcna_modules/modules_fold{fold_number}.pickle'

        the_modules, all_features = load_modules(module_file_path)

        split_module_data = split_data[list(all_features)] #maybe not necessary
        
        

        for center_method in center_methods:

            print(f'fold {fold_number} with method {center_method} started')

            normalized_data = process_data(split_module_data, center_method[:-1])

            if center_method[-1] == '0':
                opt_dim = ca.find_optimal_dimension(normalized_data)
                print(f'optimal dimension is {opt_dim}')
                center_method = center_method[:-1] + str(opt_dim)

            initial_centers = initialize_centers(the_modules, split_module_data, center_method, split_labels)

            centers = run_lbg_clustering(normalized_data, initial_centers, center_method, split_labels)

            if do_AD:
                out_directory =f'/data4/zoetis/shared/mank_experiments/{figures_or_dump}/{project_name}/AD_{center_method}_modules'
            else:
                out_directory =f'/data4/zoetis/shared/mank_experiments/{figures_or_dump}/{project_name}/{center_method}_modules'
            save_path = f'{out_directory}/modules_fold{fold_number}.pickle'

            save_modules(normalized_data, split_module_data, centers, save_path)


        fold_number += 1

