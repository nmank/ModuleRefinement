#https://github.com/ekehoe32/orthrus
#import sys
#sys.path.append('/home/katrina/a/mankovic/ZOETIS/Fall2021/Orthrus/orthrus')

import os
import numpy as np
import pandas as pd
import argparse

from orthrus.core.dataset import DataSet as DS

from orthrus.core.pipeline import *

import sys
sys.path.append('/data4/mankovic/ModuleRefinement/ModuleRefinement')
import ModuleLBG as mlbg
import center_algorithms as ca
import utils as utl


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default = 'data', type = str, help = "path to data directory")
    parser.add_argument("--refined_modules_dir", default = 'experiments/refined_modules', type = str, help = "path to refined modules directory")
    parser.add_argument("--module_types", default = ['eigengene_1_1','eigengene_2_1','eigengene_4_1','eigengene_8_1'], type = list, 
                        help = "list of the module representative types for explained variance")
    parser.add_argument("--results_file", default = 'experiments/results/evr_results.csv', type = str, help = "results file path")
    args = parser.parse_args()

    data_dir = args.data_dir
    refined_modules_dir = args.refined_modules_dir
    module_types = args.module_types
    results_file = args.results_file

    evr_results = pd.DataFrame(columns = 
                                ['Data Set', 'Center Dimension', 'Fold', 
                                'Module Number', 'Explained Variance', 'Size'])

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

    for module_type in module_types:

        module_type_dir = f'{refined_modules_dir}/{module_type}'

        central_prototype = module_type[:-4]
        data_dimension = int(module_type[-1])
        center_dimension = int(module_type[-3])

        #for folds in os.listdir(module_type_dir): #all or 5fold
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
                
                data_name = utl.shorten_data_name(dataset_name)
                ds = datasets[data_name]

                the_modules, all_features = utl.load_modules(module_path)

                module_number = 0
                for module in the_modules.iterrows():
                    mod_sig = 0
                    module_genes = module[1].item()
                    module_size = len(list(module_genes))
                    slice_dataset = ds.slice_dataset(feature_ids=list(module_genes))
                    my_mlbg = mlbg.ModuleLBG(center_method = central_prototype, center_dimension = center_dimension,
                                    data_dimension = 1, distance = 'correlation', centrality = 'degree')

                    normalized_split_data = my_mlbg.process_data(np.array(slice_dataset.data))

                    _,explained_variance_ratio = ca.eigengene(normalized_split_data, r= center_dimension, evr = True)

                    row = pd.DataFrame(data = [[dataset_name,  center_dimension, fold,
                                                 module_number, explained_variance_ratio, module_size]],
                                        columns = evr_results.columns)
                    evr_results = pd.concat([evr_results, row])

                    module_number +=1

    evr_results.to_csv(results_file)
