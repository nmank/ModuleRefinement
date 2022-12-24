#https://github.com/ekehoe32/orthrus
#import sys
#sys.path.append('/home/katrina/a/mankovic/ZOETIS/Fall2021/Orthrus/orthrus')

from torch import norm
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

import ModuleLBG as mlbg
import center_algorithms as ca

import os

import utils as utl


#import sys
#sys.path.append('/home/katrina/a/mankovic/')
#from PathwayAnalysis.SpectralClustering import SpectralClustering

from sklearn.model_selection import StratifiedKFold


def shorten_data_name(data_name):
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
    if 'ebola' in data_name:
        if 'Lethal' in data_name:
            data_name = data_name[:-7]
        if 'Tolerant' in data_name:
            data_name = data_name[:-9]      
    return data_name
      


if __name__ == '__main__':

    evr_results = pd.DataFrame(columns = 
                                ['Data Set', 'Center Dimension', 'Fold', 
                                'Module Number', 'Explained Variance', 'Size'])

    datasets = {}
    for f_name in os.listdir('./data/'):
        if 'labels' not in f_name:
            file_name = f'./data/{f_name}'
            data_name = file_name[:-4]

            data_all = pd.read_csv(file_name, index_col = 0)
            data_all.index.names = ['SampleID']
            labels_all = pd.read_csv(f'{data_name}_labels.csv', index_col = 0)
            ds = DS(data = data_all, metadata = labels_all)
            datasets[data_name[7:]] = ds


    dir_path = './refined_modules/'
    algorithm = 'WGCNA_LBG'

    module_types  = ['eigengene_1_1','eigengene_2_1','eigengene_4_1','eigengene_8_1'] #center type

    for module_type in module_types:

        module_type_dir = dir_path + module_type

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
                
                data_name = shorten_data_name(dataset_name)
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

    evr_results.to_csv('evr_results.csv')
