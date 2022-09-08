import pandas as pd
import numpy as np

import orthrus
from orthrus import core
from orthrus.core import dataset, helper

import center_algorithms as ca


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
        # the_modules = the_modules.append(row, ignore_index = True)

            
        
        helper.save_object(the_modules, save_path, overwrite=True)