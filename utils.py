import pandas as pd
import numpy as np


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

    #if 'salmonella' in file_name:
    #    labels_all.index = [str(x) for x in list(labels_all.index.astype('int'))]
    #    data_all.index = [str(x) for x in list(data_all.index.astype('int'))]

    #if 'ebola' in file_name:
    #    data_all.columns = list(data_all.columns.astype('int'))
    #    labels_all.index = ['a'+str(x) for x in list(labels_all.index.astype('int'))]
    #    data_all.index = ['a'+str(x) for x in list(data_all.index.astype('int'))]

    class_data, unique_labels = separate_classes(data_all, labels_all)

    return class_data, unique_labels, data_all, labels_all
