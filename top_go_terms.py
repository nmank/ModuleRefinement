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

def top_k(module_genes: list, organism: str, k: int) -> int:
    gp = GProfiler(return_dataframe=True)
    res = gp.profile(organism=organism,
            query=module_genes)
    query = ['GO' in s for s in res['source']]
    qres = res[query]
    if 'p_value' in qres.columns:
        top_10_rows = qres.nsmallest(k, 'p_value')
        return top_10_rows[['native', 'name', 'p_value']]
    else:
        return pd.DataFrame(columns = ['native', 'name', 'p_value'])

if __name__ == '__main__':

    go_results = pd.DataFrame(columns = 
                                ['Data Set', 'Algorithm', 'Central Prototype', 
                                'Data Dimension', 'Center Dimension', 'Fold', 
                                'Module Number', 'GO Significance'])




    central_prototype = 'module_expression'
    data_dimension = 1
    center_dimension = 1
    fold = 'all'
    dataset_name = 'gse73072_hrv_48_64_shedder48_64'

    organism = 'hsapiens'
    #organism = 'mmusculus'

    module_path = f'./refined_modules/{central_prototype}_{center_dimension}_{data_dimension}/{fold}/{dataset_name}.pickle'

    the_modules, all_features = utl.load_modules(module_path)

    best_feats = pd.DataFrame()
    #module_number = 2
    for i in range(len(the_modules)):
        module_genes = the_modules.iloc[i].item() 

        top_entries = top_k(module_genes, organism,k=10)

	
        if np.sum([('virus' in n) or ('immun' in n) for n in top_entries['name']])>0:
            print(f'module {i}')
            print(f'n genes {len(module_genes)}')
            column = pd.DataFrame(columns = [f'module {i}'], data = module_genes)
            best_feats = pd.concat([best_feats, column], axis=1)
    best_feats.to_csv(f'{dataset_name}_{central_prototype}_{center_dimension}_{data_dimension}.csv')

    print('------------------------------------------------------')

    central_prototype = 'module_expression'
    data_dimension = 1
    center_dimension = 1
    fold = 'all'
    dataset_name = 'salmonella_Liver_tolerant'

    organism = 'mmusculus'

    module_path = f'./refined_modules/{central_prototype}_{center_dimension}_{data_dimension}/{fold}/{dataset_name}.pickle'

    the_modules, all_features = utl.load_modules(module_path)

    #module_number = 2
    best_feats = pd.DataFrame()
    for i in range(len(the_modules)):
        module_genes = the_modules.iloc[i].item()

        top_entries = top_k(module_genes, organism,k=10)


        if np.sum(['immun' in n for n in top_entries['name']])>0:
            print(f'module {i}')
            print(f'n genes {len(module_genes)}')
            column = pd.DataFrame(columns = [f'module {i}'], data = module_genes)
            best_feats = pd.concat([best_feats, column], axis=1)
    best_feats.to_csv(f'{dataset_name}_{central_prototype}_{center_dimension}_{data_dimension}.csv')
