import numpy as np

import pandas as pd


import argparse

import sys
sys.path.append('/home/nmank/ModuleRefinement/ModuleRefinement')
import utils as utl



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--methods_file", 'experiments/process_results/top_go_terms.py', type = str, help = "a list of lists of parameters: [[module representative, data dimension, center dimension, fold, dataset]]")
    parser.add_argument("--results_dir", default = 'experiments/results/', type = str, help = "path to results directory")
    parser.add_argument("--refined_modules_dir", default = 'experiments/refined_modules/', type = str, help = "path to refined modules directory")
    args = parser.parse_args()

    params_file = args.methods_file
    results_dir = args.results_dir
    refined_modules_dir = args.refined_modules_dir

    params = np.array(pd.read_csv(params_file))

    go_results = pd.DataFrame(columns = 
                                ['Data Set', 'Algorithm', 'Central Prototype', 
                                'Data Dimension', 'Center Dimension', 'Fold', 
                                'Module Number', 'GO Significance'])


    for row in params:

        central_prototype = row[0]
        data_dimension = row[1]
        center_dimension = row[2]
        fold = row[3]
        dataset_name = row[4]

        if 'gse' in dataset_name:
            organism = 'hsapiens'
        else:
            organism = 'mmusculus'

        module_path = f'{refined_modules_dir}/{central_prototype}_{center_dimension}_{data_dimension}/{fold}/{dataset_name}.pickle'

        the_modules, all_features = utl.load_modules(module_path)
        print(f'modules {central_prototype}_{center_dimension}_{data_dimension} loaded')
        best_feats = pd.DataFrame()
        #module_number = 2
        for i in range(len(the_modules)):
            module_genes = the_modules.iloc[i].item() 
            
            print('.') 

            top_entries = utl.top_k(module_genes, organism,k=10)
        
            if np.sum([('immun' in n)   for n in top_entries['name']])>0:
                if 'gse' in dataset_name:
                    if np.sum([('virus' in n) for n in top_entries['name']])>0:
                        print(f'module {i}')
                        print(f'n genes {len(module_genes)}')
                        column = pd.DataFrame(columns = [f'module {i}'], data = module_genes)
                        best_feats = pd.concat([best_feats, column], axis=1)
                else:
                    print(f'module {i}')
                    print(f'n genes {len(module_genes)}')
                    column = pd.DataFrame(columns = [f'module {i}'], data = module_genes)
                    best_feats = pd.concat([best_feats, column], axis=1)
        best_feats.to_csv(f'{results_dir}/{dataset_name}_{central_prototype}_{center_dimension}_{data_dimension}.csv')

        # print('------------------------------------------------------')

        # central_prototype = 'module_expression'
        # data_dimension = 1
        # center_dimension = 1
        # fold = 'all'
        # dataset_name = 'salmonella_Liver_tolerant'

        # organism = 'mmusculus'

        # module_path = f'../refined_modules/{central_prototype}_{center_dimension}_{data_dimension}/{fold}/{dataset_name}.pickle'

        # the_modules, all_features = utl.load_modules(module_path)

        # #module_number = 2
        # best_feats = pd.DataFrame()
        # for i in range(len(the_modules)):
        #     module_genes = the_modules.iloc[i].item()

        #     top_entries = utl.top_k(module_genes, organism,k=10)


        #     if np.sum(['immun' in n for n in top_entries['name']])>0:
        #         print(f'module {i}')
        #         print(f'n genes {len(module_genes)}')
        #         column = pd.DataFrame(columns = [f'module {i}'], data = module_genes)
        #         best_feats = pd.concat([best_feats, column], axis=1)
        # best_feats.to_csv(f'../results/{dataset_name}_{central_prototype}_{center_dimension}_{data_dimension}.csv')
