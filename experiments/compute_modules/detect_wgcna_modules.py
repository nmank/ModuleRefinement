
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import argparse

import ModuleRefinement.utils as utl


'''
A script for computing WGCNA modules from data sets.

'''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default = 'data', type = str, help = "path to data directory")
    parser.add_argument("--modules_dir", default = 'experiments/modules', type = str, help = "path to modules directory")
    parser.add_argument("--results_dir", default = 'experiments/results/wgcna_rquared.csv', type = str, 
                         help = "path to results directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    modules_dir = args.modules_dir
    results_dir = args.results_dir

    prms = {}

    for file_name in os.listdir(data_dir):

        if 'label' not in file_name:

            print('------------------------')
            print(f'computing {file_name[:-4]} modules')

            if 'gse' in file_name:
                species = 'human'
            else:
               species = 'mouse'

            class_data, unique_labels, data_all, labels_all = utl.load_data(f'{data_dir}/{file_name}')

            out_file = f'{modules_dir}/all/{file_name[:-4]}.pickle'
            prms[file_name[:-4]] = utl.wgcna_modules(data_all, species, out_file)

            for dta, lbl in zip(class_data, unique_labels):
                out_file = f'{modules_dir}/all/{file_name[:-4]}_{lbl}.pickle'
                prms[file_name[:-4]+'_'+str(lbl)] = utl.wgcna_modules(dta, species, out_file)

            print(f'computing 5 fold modules...')

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            skf.get_n_splits(data_all, labels_all)

            fold_number = 0 
            for train_index, test_index in skf.split(data_all, labels_all):
                split_data = data_all.iloc[train_index]
                split_labels = labels_all.iloc[train_index]

                out_file = f'{modules_dir}/5fold/fold{fold_number}_{file_name[:-4]}.pickle'
                prms[str(fold_number) + '_' + file_name[:-4]] = utl.wgcna_modules(split_data, species, out_file)

                class_data, unique_labels = utl.separate_classes(split_data, split_labels)

                for dta, lbl in zip(class_data, unique_labels):
                    out_file = f'{modules_dir}/5fold/fold{fold_number}_{file_name[:-4]}_{lbl}.pickle'
                    prms[str(fold_number) + '_' + file_name[:-4]+'_'+str(lbl)] = utl.wgcna_modules(dta, species, out_file)

                fold_number += 1

    wgcna_rsquared = pd.DataFrame.from_dict(prms, orient="index")
    wgcna_rsquared.to_csv(results_dir)
