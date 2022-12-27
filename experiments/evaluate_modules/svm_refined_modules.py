#https://github.com/ekehoe32/orthrus
#import sys
#sys.path.append('/home/katrina/a/mankovic/ZOETIS/Fall2021/Orthrus/orthrus')

import numpy as np


import pandas as pd

from orthrus.core.dataset import DataSet as DS

import os

import utils as utl

import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default = '/home/nmank/ModuleRefinement/data', type = str, help = "path to data directory")
    parser.add_argument("--refined_modules_dir", default = '/home/nmank/ModuleRefinement/experiments/refined_modules', type = str, help = "path to refined modules directory")
    parser.add_argument("--results_file", default = '/home/nmank/ModuleRefinement/experiments/results/svm_lbg_score.csv', type = str, help = "path to results file")
    args = parser.parse_args()

    data_dir = args.data_dir
    modules_dir = args.modules_dir
    refined_modules_dir = args.refined_modules_dir
    results_file = args.results_file

    svm_results = pd.DataFrame(columns = 
                                ['Data Set', 'Algorithm', 'Central Prototype', 
                                'Data Dimension', 'Center Dimension', 'Fold', 
                                'Module Number', 'BSR', 'Size'])

    datasets = {}
    for f_name in os.listdir(data_dir):
        if 'labels' not in f_name:
            file_name = f'{data_dir}/{f_name}'
            data_name = file_name[:-4]

            data_all = pd.read_csv(file_name, index_col = 0)
            data_all.index.names = ['SampleID']
            labels_all = pd.read_csv(f'{data_name}_labels.csv', index_col = 0)
            ds = DS(data = data_all, metadata = labels_all)
            datasets[data_name[7:]] = ds


    algorithm = 'WGCNA_LBG'

    for module_type in os.listdir(refined_modules_dir): #center type

        module_type_dir = f'{refined_modules_dir}/{module_type}'

        central_prototype = module_type[:-4]
        data_dimension = module_type[-1]
        center_dimension = module_type[-3]

        # for folds in os.listdir(module_type_dir): #all or 5fold
        for folds in ['5fold']: #5fold    
            folder_path = f'{module_type_dir}/{folds}'

            for dataset_name in os.listdir(folder_path):
                module_path = f'{folder_path}/{dataset_name}'


                if 'fold' in dataset_name:
                    fold = dataset_name[4]
                    dataset_name = dataset_name[6:-7]
                else:
                    dataset_name = dataset_name[:-7]
                    fold = 'all'
                
                if 'gse' in dataset_name:
                    organism = 'human'
                else:
                    organism = 'mouse'

                if ('True' not in dataset_name) and ('False' not in dataset_name):
                
                    data_name = utl.shorten_data_name(dataset_name)
                    ds = datasets[data_name]

                    the_modules, all_features = utl.load_modules(module_path)

                    module_number = 0
                    for module in the_modules.iterrows():
                        mod_sig = 0
                        module_genes = module[1].item()
                        module_size = len(list(module_genes))
                        slice_dataset = ds.slice_dataset(feature_ids=list(module_genes))

                        try:
                            bsr = utl.loso_test(slice_dataset, fold)
                        except ValueError:
                            bsr = np.nan   
                                
                        row = pd.DataFrame(columns = list(svm_results.columns),
                                            data = [[dataset_name, algorithm, central_prototype,
                                                    data_dimension, center_dimension, fold,
                                                    module_number, bsr, module_size]])
                        svm_results = svm_results.append(row, ignore_index = True)
                        module_number +=1

    svm_results.to_csv(results_file)


