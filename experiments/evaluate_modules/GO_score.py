#import sys

#sys.path.append('/home/katrina/a/mankovic/FlagIRLS')
import sys
sys.path.append('./ModuleRefinement')
import center_algorithms as ca

import orthrus
from orthrus import core
from orthrus.core import dataset, helper
import numpy as np

from matplotlib import pyplot as plt 

import gseapy as gp

import pandas as pd

import plotly.express as px
import os

import argparse

import utils as utl



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default = '/home/nmank/ModuleRefinement/data', type = str, help = "path to data directory")
    parser.add_argument("--modules_dir", default = '/home/nmank/ModuleRefinement/experiments/modules', type = str, help = "path to modules directory")
    parser.add_argument("--refined_modules_dir", default = '/home/nmank/ModuleRefinement/experiments/refined_modules', type = str, help = "path to refined modules directory")
    parser.add_argument("--results_file", default = '/home/nmank/ModuleRefinement/experiments/results/go_score.csv', type = str, help = "path to results file")
    args = parser.parse_args()

    data_dir = args.data_dir
    modules_dir = args.modules_dir
    refined_modules_dir = args.refined_modules_dir
    results_file = args.results_file


    go_results = pd.DataFrame(columns = 
                                ['Data Set', 'Algorithm', 'Central Prototype', 
                                'Data Dimension', 'Center Dimension', 'Fold', 
                                'Module Number', 'GO Significance'])


    algorithm = 'WGCNA_LBG'

    for module_type in os.listdir(refined_modules_dir): #center type

        module_type_dir = f'{refined_modules_dir}/{module_type}'

        central_prototype = module_type[:-4]
        data_dimension = module_type[-1]
        center_dimension = module_type[-3]

        for folds in os.listdir(module_type_dir): #all or 5fold
            
            folder_path = f'{module_type_dir}/{folds}'

            for dataset_name in os.listdir(folder_path):
                if ('ebola' not in dataset_name) and ('Spleen' not in dataset_name):
                    module_path = f'{folder_path}/{dataset_name}'

                    if 'fold' in dataset_name:
                        fold = dataset_name[4]
                        dataset_name = dataset_name[6:-7]
                    else:
                        dataset_name = dataset_name[:-7]
                        fold = 'all'
                
                    if 'gse' in dataset_name:
                        organism = 'hsapiens'
                    else:
                        organism = 'mmusculus'

                    the_modules, all_features = utl.load_modules(module_path)

                    module_number = 0
                    for module in the_modules.iterrows():
                        mod_sig = 0
                        module_genes = module[1].item()   
                        
                        try:
                            mod_sig = utl.module_significance(module_genes, organism)
                    
                    
                            row = pd.DataFrame(columns = list(go_results.columns),
                                        data = [[dataset_name, algorithm, central_prototype,
                                                 data_dimension, center_dimension, fold,
                                                 module_number, mod_sig]])
                            go_results = pd.concat([go_results,row])

                        except AssertionError:
                            print('bad gateway error?')

                        module_number +=1


    algorithm = 'WGCNA'
    

    for folds in os.listdir(modules_dir): #all or 5fold
        
        folder_path = f'{modules_dir}/{folds}'

        for dataset_name in os.listdir(folder_path):
            if ('ebola' not in dataset_name) and ('Spleen' not in dataset_name):
                module_path = f'{folder_path}/{dataset_name}'

                if 'fold' in dataset_name:
                    fold = dataset_name[4]
                    dataset_name = dataset_name[6:-7]
                else:
                    dataset_name = dataset_name[:-7]
                    fold = 'all'
            
                if 'gse' in dataset_name:
                    organism = 'hsapiens'
                else:
                    organism = 'mmusculus'

                the_modules, all_features = utl.load_modules(module_path)

                module_number = 0
                for module in the_modules.iterrows():
                    mod_sig = 0
                    module_genes = module[1].item()   
                    
                    try:
                        mod_sig = utl.module_significance(module_genes, organism)
                
                
                        row = pd.DataFrame(columns = list(go_results.columns),
                                    data = [[dataset_name, algorithm, '',
                                                '', '', fold,
                                                module_number, mod_sig]])
                        go_results = pd.concat([go_results,row])
                    except AssertionError:
                        print('bad gateway error?')

                    module_number +=1


    go_results.to_csv(results_file)
