
import pandas as pd
import numpy as np

from orthrus.core import helper

import os

import pickle

import PyWGCNA

from sklearn.model_selection import StratifiedKFold

def separate_classes(data_all, labels_all):
    unique_labels = np.unique(labels_all['label'])

    class_data = []
    for label in unique_labels:
        idx = labels_all['label'] == label
        label_ids = labels_all[idx].index
        
        class_data.append(data_all.loc[label_ids])

    return class_data, unique_labels

def load_data(file_name):
    data_all = pd.read_csv(file_name, index_col = 0)
    data_all.index.names = ['SampleID']
    labels_all = pd.read_csv(f'{file_name[:-4]}_labels.csv', index_col = 0)

    #if 'salmonella' in file_name:
    #    labels_all.index = [str(x) for x in list(labels_all.index.astype('int'))]
    #    data_all.index = [str(x) for x in list(data_all.index.astype('int'))]

    #if 'ebola' in file_name:
    #    data_all.columns = list(data_all.columns.astype('int'))
    #    labels_all.index = ['a'+str(x) for x in list(labels_all.index.astype('int'))]
    #    data_all.index = ['a'+str(x) for x in list(data_all.index.astype('int'))]

    class_data, unique_labels = separate_classes(data_all, labels_all)

    return class_data, unique_labels, data_all, labels_all

def wgcna_modules(split_data, species, save_path):
    split_data_wgcna = split_data.copy()
    split_data_wgcna.reset_index(inplace=True)
    
    #split_data.to_csv('temp.csv')
    #geneExp = 'temp.csv'
    #import IPython; IPython.embed()
    pyWGCNA_Z75 = PyWGCNA.WGCNA(name='Z75', species=species, geneExp=split_data_wgcna, save=True)

    pyWGCNA_Z75.preprocess()
    rcut_val = .9
    for i in range(1,6):
        try:
            print(f'attempt with {i}')
            pyWGCNA_Z75.findModules()
        except:
            rcut_val = .9-.1*i
            print(f'setting RsquaredCut to {rcut_val}')
            pyWGCNA_Z75.RsquaredCut = rcut_val
            #pyWGCNA_Z75.findModules()
        else:
            break

    # the_modules = pd.DataFrame(pyWGCNA_Z75.datExpr.var['moduleLabels'])

    # the_modules.to_csv(f'modules_fold{fold_number}.csv')

    the_modules = pd.DataFrame(columns = ['Feature Set'])
    for lbl in np.unique(pyWGCNA_Z75.datExpr.var['moduleLabels']):
        idx = pyWGCNA_Z75.datExpr.var['moduleLabels'] == lbl
        genes_in_module = list(pyWGCNA_Z75.datExpr.var['moduleLabels'][idx].index)
        row = pd.DataFrame(columns = ['Feature Set'], data = [[genes_in_module]])
        the_modules = the_modules.append(row, ignore_index = True)

    helper.save_object(the_modules, save_path, overwrite=False)
    return rcut_val


if __name__ == '__main__':


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

            class_data, unique_labels, data_all, labels_all = load_data(data_dir +file_name)

            out_file = f'./modules/all/{file_name}'
            prms[file_name[:-4]] = wgcna_modules(data_all, species, out_file)

            for dta, lbl in zip(class_data, unique_labels):
                out_file = f'./modules/all/{file_name[:-4]}_{lbl}.csv'
                prms[file_name[:-4]] = wgcna_modules(dta, species, out_file)

            print(f'computing 5 fold modules...')

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            skf.get_n_splits(data_all, labels_all)

            fold_number = 0
            for train_index, test_index in skf.split(data_all, labels_all):
                split_data = data_all.iloc[train_index]
                split_labels = labels_all.iloc[train_index]

                out_file = f'./modules/5fold/fold{fold_number}_{file_name}'
                prms[file_name[:-4]] = wgcna_modules(split_data, species, out_file)

                class_data, unique_labels = separate_classes(split_data, split_labels)

                for dta, lbl in zip(class_data, unique_labels):
                    out_file = f'./modules/5fold/fold{fold_number}_{file_name[:-4]}_{lbl}.csv'
                    prms[file_name[:-4]] = wgcna_modules(dta, species, out_file)

                fold_number += 1

    wgcna_rsquared = pd.DataFrame.from_dict(prms, orient="index")
    df.to_csv("wgcna_rquared.csv")
