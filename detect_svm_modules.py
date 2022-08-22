import pandas as pd
import numpy as np

import orthrus
from orthrus import core
from orthrus.core import dataset, helper
import pickle
from sklearn.preprocessing import StandardScaler, FunctionTransformer

import sys
sys.path.append('/home/katrina/a/mankovic/')
from PathwayAnalysis.SpectralClustering import SpectralClustering

from sklearn.model_selection import StratifiedKFold

def run_spectral_clustering(compare: str, seqID_labels: list, sorted_X: np.ndarray, 
                            loo: bool = True, fiedler: bool = True) -> tuple:                           
    my_sc = SpectralClustering(similarity = compare)

    binary_labels = np.zeros(len(seqID_labels))
    AD_idx = np.where(np.array(seqID_labels) == 'AD')[0]
    binary_labels[AD_idx] = 1
    my_sc.fit(sorted_X, binary_labels)

    # print(my_sc.transform(sorted_X, binary_labels, loo = loo, fiedler = fiedler))
    nodes, bsrs = my_sc.transform(sorted_X, binary_labels, loo = loo, fiedler = fiedler)[:2]

    return nodes, bsrs

if __name__ == '__main__':

    for similarity in ['heatkernel', 'zobs', 'correlation']:

        ds = dataset.load_dataset('/data4/zoetis/Data/TPM_C1_Z34_Z40_Z42_Z75.ds')
        sample_ids  = ds.metadata['Project'] == 'Z75'

        all_features = pd.read_csv('/data4/zoetis/Decks/June_2022/results/z75_cross_validated_feature.csv', index_col=0)

        Z75_dataset = ds.slice_dataset(sample_ids=sample_ids)

        Z75_data = Z75_dataset.data

        labels = Z75_dataset.metadata['Diagnosis']

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        skf.get_n_splits(Z75_data, labels)

        fold_number = 0
        for train_index, test_index in skf.split(Z75_data, labels):
            print(f'fold {fold_number} started')
            rand_IDs = list(all_features[all_features[str(fold_number)]].index)

            Z75_dataset = ds.slice_dataset(sample_ids=sample_ids, feature_ids = rand_IDs)

            Z75_data = Z75_dataset.data

            labels = Z75_dataset.metadata['Diagnosis'][train_index]

            split_data = Z75_data.iloc[train_index]

            # split_data = FunctionTransformer(np.log2).fit_transform(split_data)

            split_data = StandardScaler().fit_transform(split_data)

            nodes, bsrs = run_spectral_clustering(similarity, labels, split_data, loo = True, fiedler = True)

            the_modules = pd.DataFrame(columns = ['Feature Set'])
            for nds in nodes:
                genes_in_module = [rand_IDs[n] for n in nds]
                row = pd.DataFrame(columns = ['Feature Set'], data = [[genes_in_module]])
                the_modules = the_modules.append(row, ignore_index = True)

            
            save_path = f'/data4/zoetis/shared/mank_experiments/figures/Z75/svm_modules/modules_{similarity}_fold{fold_number}.pickle'
            helper.save_object(the_modules, save_path, overwrite=True)


            fold_number += 1

