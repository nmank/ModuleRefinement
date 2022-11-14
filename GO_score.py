#import sys

#sys.path.append('/home/katrina/a/mankovic/FlagIRLS')

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

from gprofiler import GProfiler

import utils as utl

def module_significance(module_genes: list, organism: str) -> int:
    gp = GProfiler(return_dataframe=True)
    res = gp.profile(organism=organism,
            query=module_genes)
    query = (res['p_value'] < .05) &\
            (['GO' in s for s in res['source']])
    mod_sig = np.sum(-np.log10(np.array(res[query]['p_value'])))
    return mod_sig


if __name__ == '__main__':

    go_results = pd.DataFrame(columns = 
                                ['Data Set', 'Algorithm', 'Central Prototype', 
                                'Data Dimension', 'Center Dimension', 'Fold', 
                                'Module Number', 'GO Significance'])


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
                if 'ebola' not in dataset_name:
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
                            mod_sig = module_significance(module_genes, organism)
                    
                    
                            row = pd.DataFrame(columns = list(go_results.columns),
                                        data = [[dataset_name, algorithm, central_prototype,
                                                 data_dimension, center_dimension, fold,
                                                 module_number, mod_sig]])
                            go_results = pd.concat([go_results,row])

                        except AssertionError:
                            print('bad gateway error?')

                        module_number +=1


    module_type_path = './modules/'
    algorithm = 'WGCNA'
    

    for folds in os.listdir(module_type_dir): #all or 5fold
        
        folder_path = f'{module_type_dir}/{folds}'

        for dataset_name in os.listdir(folder_path):
            if 'ebola' not in dataset_name:
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
                        mod_sig = module_significance(module_genes, organism)
                
                
                        row = pd.DataFrame(columns = list(go_results.columns),
                                    data = [[dataset_name, algorithm, '',
                                                '', '', fold,
                                                module_number, mod_sig]])
                        go_results = pd.concat([go_results,row])
                    except AssertionError:
                        print('bad gateway error?')

                    module_number +=1


    go_results.to_csv('GO_score.csv')
