#https://github.com/ekehoe32/orthrus
import sys
sys.path.append('/home/katrina/a/mankovic/ZOETIS/Fall2021/Orthrus/orthrus')

import orthrus
from orthrus import core
from orthrus.core import dataset, helper
from sklearn.model_selection import LeaveOneOut
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import pickle

import pandas as pd

from orthrus.core.dataset import DataSet as DS

from orthrus.core.pipeline import *
from sklearn.preprocessing import FunctionTransformer
from orthrus.preprocessing.imputation import HalfMinimum
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

import os

import utils as utl


import sys
sys.path.append('/home/katrina/a/mankovic/')
from PathwayAnalysis.SpectralClustering import SpectralClustering

from sklearn.model_selection import StratifiedKFold




def loso_test(ds, fold_number):

    supervised_attr = 'Diagnosis'

    partitioner = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    partition_5fold = Partition(process=partitioner,
                            process_name='partition_5fold',
                            verbosity=2,
                            split_attr = supervised_attr)

    svm_model = Classify(process=LinearSVC(dual=False),
                process_name='SVC',
                class_attr=supervised_attr,
                verbosity=1)

    # log2 transformation
    log2 = Transform(process=FunctionTransformer(np.log2),
                    process_name='log2',
                    retain_f_ids=True)

    # half-minimum imputation
    hf = Transform(process=HalfMinimum(missing_values=0),
                process_name='half-min',
                retain_f_ids=True)

    
    std = Transform(process=StandardScaler(),
                    process_name='std',
                    retain_f_ids=True)


    # define balance accuracy score process
    scorer = balanced_accuracy_score
    bsr = Score(process=scorer,
            process_name='bsr',
            pred_attr=supervised_attr,
            verbosity=0)

    processes=(partition_5fold, hf, log2, std, svm_model, bsr)

    pipeline = Pipeline(processes=processes)

    pipeline.run(ds, checkpoint=False)


    fold_test_bsr = bsr.collapse_results()['class_pred_scores'].loc['Test'].loc[f'batch_{fold_number}']
    # train_scores = scores.loc['Train']

    return fold_test_bsr


if __name__ == '__main__':

    svm_results = pd.DataFrame(columns = 
                                ['Data Set', 'Algorithm', 'Central Prototype', 
                                'Data Dimension', 'Center Dimension', 'Fold', 
                                'Module Number', 'BSR'])

    datasets = {}
    for f_name in os.listdir('./data/'):
        if 'labels' not in f_name:
            file_name = f'./data/{f_name}'
            data_name = file_name[:-4]

            data_all = pd.read_csv(file_name, index_col = 0)
            data_all.index.names = ['SampleID']
            labels_all = pd.read_csv(f'{data_name}_labels.csv', index_col = 0)
            ds = DS(data = data_all, metadata = labels_all)
            datasets[data_name] = ds
    


    dir_path = './refined_modules/'
    algorithm = 'WGCNA_LBG'

    for module_type in os.listdir(dir_path): #center type

        module_type_dir = dir_path + module_type

        central_prototype = module_type[:-4]
        data_dimension = module_type[-1]
        center_dimension = module_type[-3]

        for folds in os.listdir(module_type_dir): #all or 5fold
            
            folder_path = f'{module_type_dir}/{folds}'

            for dataset_name in os.listdir(folder_path):
                module_path = f'{folder_path}/{dataset_name}'

                if 'fold' in dataset_name:
                    dataset_name = dataset_name[6:-7]
                    fold = dataset_name[4]
                else:
                    dataset_name = dataset_name[:-7]
                    fold = 'all'
                
                if 'gse' in dataset_name:
                    organism = 'human'
                else:
                    organism = 'mouse'

                ds = datasets[dataset_name]

                the_modules, all_features = utl.load_modules(module_path)

                module_number = 0
                for module in the_modules.iterrows():
                    mod_sig = 0
                    module_genes = module[1].item()

                    slice_dataset = ds.slice_dataset(feature_ids=list(module_genes))

                    try:
                        bsr = loso_test(slice_dataset, fold)
                    except ValueError:
                        bsr = np.nan   
                             
                    row = pd.DataFrame(columns = list(svm_results.columns),
                                        data = [[dataset_name, algorithm, central_prototype,
                                                 data_dimension, center_dimension, fold,
                                                 module_number, bsr]])
                    svm_results = svm_results.append(row, ignore_index = True)

                    module_number +=1


    module_type_path = './modules/'
    algorithm = 'WGCNA'
    

    for folds in os.listdir(module_type_dir): #all or 5fold
        
        folder_path = f'{module_type_dir}/{folds}'

        for dataset_name in os.listdir(folder_path):
            module_path = f'{folder_path}/{dataset_name}'

            if 'fold' in dataset_name:
                dataset_name = dataset_name[6:-7]
                fold = dataset_name[4]
            else:
                dataset_name = dataset_name[:-7]
                fold = 'all'
            
            if 'gse' in dataset_name:
                organism = 'human'
            else:
                organism = 'mouse'

            the_modules, all_features = utl.load_modules(module_path)

            module_number = 0
            for module in the_modules.iterrows():
                mod_sig = 0
                module_genes = module[1].item()   

                slice_dataset = ds.slice_dataset(feature_ids=list(module_genes))

                try:
                    bsr = loso_test(slice_dataset, fold)
                except ValueError:
                    bsr = np.nan   
                            
                row = pd.DataFrame(columns = list(svm_results.columns),
                                    data = [[dataset_name, algorithm, central_prototype,
                                                data_dimension, center_dimension, fold,
                                                module_number, bsr]])
                svm_results = svm_results.append(row, ignore_index = True)

                module_number +=1


    svm_results.to_csv(f'svm_score.csv')

