import pandas as pd
import numpy as np

from gprofiler import GProfiler
import PyWGCNA
import ModuleLBG as mlbg

import os

import orthrus
from orthrus import core
from orthrus.core import dataset, helper


def separate_classes(data_all: pd.DataFrame, labels_all: pd.DataFrame) -> tuple:
    unique_labels = np.unique(labels_all['label'])

    class_data = []
    for label in unique_labels:
        idx = labels_all['label'] == label
        label_ids = labels_all[idx].index
        
        class_data.append(data_all.loc[label_ids])

    return class_data, unique_labels

def load_data(file_name: str) -> tuple:
    data_all = pd.read_csv(file_name, index_col = 0)
    data_all.index.names = ['SampleID']
    labels_all = pd.read_csv(f'{file_name[:-4]}_labels.csv', index_col = 0)

    class_data, unique_labels = separate_classes(data_all, labels_all)

    return class_data, unique_labels, data_all, labels_all

def load_modules(file_path: str) -> tuple:
    the_modules = helper.load_object(file_path)
    all_features = set()
    for _, row in the_modules.iterrows():
        all_features = set(row.item()).union(all_features)

    return the_modules, all_features

def save_modules(split_module_data: pd.DataFrame, labels: np.array, save_path: str) -> None:
    the_modules = pd.DataFrame(columns = ['Feature Set'])
    

    for module_number in list(np.unique(labels)):
        genes_in_one_module = list(split_module_data.T[labels == module_number].index)
        row = pd.DataFrame(columns = ['Feature Set'], data = [[genes_in_one_module]])
        the_modules = pd.concat([the_modules, row])
        helper.save_object(the_modules, save_path, overwrite=True)

def wgcna_modules(split_data: pd.DataFrame, species: str, save_path: str) -> float:
    split_data_wgcna = split_data.copy()
    split_data_wgcna.index = [str(s) for s in split_data_wgcna.index]
    split_data_wgcna.reset_index(inplace=True)
     
    #split_data.to_csv('temp.csv')
    #geneExp = 'temp.csv'
    pyWGCNA_Z75 = PyWGCNA.WGCNA(name='Z75', species=species, geneExp=split_data_wgcna, save=True)

    pyWGCNA_Z75.preprocess()
    rcut_val = .9
    running = True
    while running:
        try:
            pyWGCNA_Z75.findModules()
            running = False
        except:
            rcut_val =rcut_val - .1
            print(f'setting RsquaredCut to {rcut_val}')
            pyWGCNA_Z75.RsquaredCut = rcut_val
    pyWGCNA_Z75.findModules()

    # the_modules = pd.DataFrame(pyWGCNA_Z75.datExpr.var['moduleLabels'])

    # the_modules.to_csv(f'modules_fold{fold_number}.csv')

    the_modules = pd.DataFrame(columns = ['Feature Set'])
    for lbl in np.unique(pyWGCNA_Z75.datExpr.var['moduleLabels']):
        idx = pyWGCNA_Z75.datExpr.var['moduleLabels'] == lbl
        genes_in_module = list(pyWGCNA_Z75.datExpr.var['moduleLabels'][idx].index)
        row = pd.DataFrame(columns = ['Feature Set'], data = [[genes_in_module]])
        the_modules = the_modules.append(row, ignore_index = True)

    #uncomment the line below	
    helper.save_object(the_modules, save_path, overwrite=False)
    return rcut_val

def refined_modules(split_data: pd.DataFrame, module_path: str, center_methods: list = [],
                    save_path_prefix: str = 'experiments/refined_modules'):
    split_path = module_path.split('/')

    the_modules, _ = load_modules(module_path)
    
    for center_method in center_methods:
        center_method_str = f'{center_method[0]}_{center_method[1]}_{center_method[2]}'
        save_path0 = f'{save_path_prefix}/{center_method_str}'
        if not os.path.isdir(save_path0):
            os.mkdir(save_path0)
        
        save_path1 = f'{save_path0}/{split_path[2]}'
        if not os.path.isdir(save_path1):
            os.mkdir(save_path1)

        save_path =  f'{save_path1}/{split_path[3][:-7]}.pickle'

        split_data = split_data.loc[:, (split_data != 0).any(axis=0)] #remove columns with all 0s

        #make index here! this is who's in what module
        feature_names = list(split_data.columns)
        feature_labels = pd.DataFrame(columns = feature_names, data = np.zeros((1,len(feature_names))))
        ii=0
        for _, m in the_modules.iterrows():
            feature_labels[m] = ii
            ii+=1
        index = feature_labels.iloc[0]

        restricted_data = split_data[feature_names]

        if center_method[2] > 1:
            my_mlbg = mlbg.ModuleLBG(center_method = center_method[0], center_dimension = center_method[1],
                                 data_dimension = center_method[2], distance = 'max correlation', centrality = 'degree')
        else:
            my_mlbg = mlbg.ModuleLBG(center_method = center_method[0], center_dimension = center_method[1],
                                 data_dimension = center_method[2], distance = 'correlation', centrality = 'degree')

        normalized_split_data = my_mlbg.process_data(np.array(restricted_data))

        my_mlbg.calc_centers(normalized_split_data, index)


        my_mlbg.fit_transform(normalized_split_data)

        labels = my_mlbg.get_labels(normalized_split_data)

        #only save if it converged
        if my_mlbg.errs_[-1] <= my_mlbg.epsilon_:
            save_modules(restricted_data, labels, save_path)

def module_significance(module_genes: list, organism: str) -> int:
    gp = GProfiler(return_dataframe=True)
    res = gp.profile(organism=organism,
            query=module_genes)
    query = (res['p_value'] < .05) &\
            (['GO' in s for s in res['source']])
    mod_sig = np.sum(-np.log10(np.array(res[query]['p_value'])))
    return mod_sig

def shorten_data_name(data_name: str):
    if 'gse' in data_name:
        if 'control' in data_name:
            data_name = data_name[:-8]
        if 'shedder48_64' in data_name:
            data_name = data_name[:-13]
    if 'salmonella' in data_name:
        if 'tolerant' in data_name:
            data_name = data_name[:-9]
        if 'susceptible' in data_name:
            data_name = data_name[:-12]   
    return data_name 