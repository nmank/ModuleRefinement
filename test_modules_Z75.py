#https://github.com/ekehoe32/orthrus
import sys
sys.path.append('/home/katrina/a/mankovic/ZOETIS/Fall2021/Orthrus/orthrus')

import orthrus
from orthrus import core
from orthrus.core import dataset, helper
from sklearn.model_selection import LeaveOneOut
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import pickle

import pandas
from orthrus.core.pipeline import *
from sklearn.preprocessing import FunctionTransformer
from orthrus.preprocessing.imputation import HalfMinimum
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

import os


import sys
sys.path.append('/home/katrina/a/mankovic/')
from PathwayAnalysis.SpectralClustering import SpectralClustering

from sklearn.model_selection import StratifiedKFold




def loso_test(ds, fold_number):

    supervised_attr = 'Diagnosis'

    partitioner = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    partition_5fold = Partition(process=partitioner,
                            process_name='partition_loo',
                            verbosity=2,
                            split_attr = supervised_attr)

    svm_model = Classify(process=LinearSVC(dual=False),
                process_name='SVC',
                class_attr=supervised_attr,
                verbosity=1)

    # log2 transformation
    log2 = Transform(process=FunctionTransformer(np.log2),
                    process_name='log2',
                    retain_f_ids=True)

    # half-minimum imputation
    hf = Transform(process=HalfMinimum(missing_values=0),
                process_name='half-min',
                retain_f_ids=True)

    
    std = Transform(process=StandardScaler(),
                    process_name='std',
                    retain_f_ids=True)


    # define balance accuracy score process
    scorer = balanced_accuracy_score
    bsr = Score(process=scorer,
            process_name='bsr',
            pred_attr=supervised_attr,
            verbosity=0)

    processes=(partition_5fold, hf, log2, std, svm_model, bsr)

    pipeline = Pipeline(processes=processes)

    pipeline.run(ds, checkpoint=False)


    fold_test_bsr = bsr.collapse_results()['class_pred_scores'].loc['Test'].loc[f'batch_{fold_number}']
    # train_scores = scores.loc['Train']

    return fold_test_bsr



def plot_it(ds, attr, base_dir, save_name):
    pca = PCA(n_components=2, whiten=False)
    ds.path = base_dir
    ds.visualize(embedding=pca,
        attr=attr,
        title='',
        subtitle='',
        palette='bright',
        alpha=.75,
        save=True,
        save_name=save_name)

if __name__ == '__main__':

    results = pandas.DataFrame(columns = ['featureset type', 'fold number', 'similarity', 'module number', 'number of features', 'bsr'])

    AD_string = 'AD_'
    # methods = ['wgcna']
    methods = ['wgcna',
              'eigengene1', 
              'flag_mean1', 
              'flag_median1', 
              'eigengene4',
              'flag_mean4',
              'flag_median4',
              'eigengene8',
              'flag_mean8',
              'flag_median8']

    figures_or_dump = 'dump'

    for method in methods:
        print(f'starting {method} modules')

        if method == 'svm':
            similarities = ['correlation', 'heatkernel', 'zobs']
        else:
            similarities = ['']
        
        for similarity in similarities:
            print(f'similarity {similarity}')

            for fold_number in range(5):
                print(f'fold number {fold_number}')

                if len(similarity) > 1:
                    featuresets = helper.load_object(f'/data4/zoetis/shared/mank_experiments/{figures_or_dump}/Z75/{AD_string}{method}_modules/modules_{similarity}_fold{fold_number}.pickle')
                else:
                    featuresets = helper.load_object(f'/data4/zoetis/shared/mank_experiments/{figures_or_dump}/Z75/{AD_string}{method}_modules/modules_fold{fold_number}.pickle')

                module_number = 0
                for _, featureset_series in featuresets.iterrows():
                    print(f'module number {module_number}')
                    featureset = featureset_series.item()

                    ds = dataset.load_dataset(os.path.join('/data4/zoetis/Data/TPM_C1_Z34_Z40_Z42_Z75.ds'))
                    sample_ids  = ds.metadata['Project'] == 'Z75'

                    Z75_dataset = ds.slice_dataset(sample_ids=sample_ids, feature_ids=featureset)


                    try:
                        bsr = loso_test(Z75_dataset, fold_number)
                    except ValueError:
                        bsr = np.nan

                    new_row = pandas.DataFrame(columns = ['featureset type', 'fold number', 'similarity', 'module number', 'number of features', 'bsr'], data = [[method, fold_number, similarity, module_number, len(featureset), bsr]])

                    results = results.append(new_row, ignore_index = True)
                    module_number +=1







                    
    print(results)
    results.to_csv(f'../{figures_or_dump}/Z75/Z75_{AD_string}module_5fold_test_results.csv')
