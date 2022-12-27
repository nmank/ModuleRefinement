import numpy as np
import pandas as pd

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--wgcna_file", default = 'experiments/results/svm_wgcna_score.csv', type = str, help = "path to wgcna svm results file")
    parser.add_argument("--lbg_file", default = 'experiments/results/svm_lbg_score.csv', type = str, help = "path to lbg go results file")
    parser.add_argument("--center_methods_file", default = 'experiments/compute_modules/center_methods.csv', type = str, 
                        help = "path to csv file with 3 columns: 1) center type, 2) center dimension, 3) data dimension")
    parser.add_argument("--results_dir", default = 'experiments/results/', type = str, help = "path to results directory")
    args = parser.parse_args()

    wgcna_file = args.wgcna_file
    lbg_file = args.lbg_file
    center_methods_file = args.center_methods_file
    results_dir = args.results_dir


    wgcna_svm = pd.read_csv(wgcna_file, index_col = 0)

    lbg_svm = pd.read_csv(lbg_file, index_col = 0)
    
    center_methods = np.array(pd.read_csv(center_methods_file, index_col = 0))


    data_sets = np.unique(lbg_svm["Data Set"])

    svm_better = pd.DataFrame(columns = ['Relative Gain', 'Fold', 'Prototype', 'Data Dimension', 'Center Dimension', 'Data Set'])
    for data_set in data_sets:
            for prototype, center_dim, data_dim in center_methods:
                    for fold in range(5):
                            wgcna_query = (wgcna_svm["Algorithm"] == 'WGCNA') &\
                                        (wgcna_svm["Fold"] == fold) &\
                                        (wgcna_svm["Data Set"] == data_set)
                            if np.sum(wgcna_query) > 0:
                                    wgcna_total = wgcna_svm[wgcna_query]["BSR"].sum()
                                    query = (lbg_svm["Algorithm"] == 'WGCNA_LBG') &\
                                            (lbg_svm["Central Prototype"] == prototype) &\
                                            (lbg_svm["Data Dimension"] == data_dim) &\
                                            (lbg_svm["Center Dimension"] == center_dim) &\
                                            (lbg_svm["Fold"] == fold) &\
                                            (lbg_svm["Data Set"] == data_set) &\
                                            (lbg_svm["Size"] <= 22276)
                                    if np.sum(query) > 0:
                                            eigengene_total = lbg_svm[query]["BSR"].sum()
                                            # print((mean_feats['wgcna']/mean_feats[prototype]))
                                            relative_gain = (eigengene_total/(wgcna_total+.0001))-1
                                            row = pd.DataFrame(columns = svm_better.columns,
                                                            data = [[relative_gain, fold, prototype, data_dim, center_dim, data_set]])
                                            svm_better = svm_better.append(row, ignore_index = True)

    svm_better.to_csv(f'{results_dir}/refined_svm_score.csv')

    svm_all = svm_better.copy()

    data_sets = np.unique(svm_all['Data Set'])
    data_mask = ['ebola' not in d for d  in data_sets]
    data_sets = data_sets[data_mask]
    idx = [w in data_sets for w in svm_all['Data Set']]
    svm_all = svm_all[idx]


    #make a data frame with mean and standard deviation for each method
    mean_svm_data = pd.DataFrame(columns = ['Mean Relative Gain', 'Std Relative Gain', 'Center Dimension', 
                                            'Data Dimension', 'Prototype', 'Data Set'])
    simple_mean_svm_data = pd.DataFrame(columns = ['Data Set', 'Module Refinement Method', 'Mean Relative Gain'])
    for ds in data_sets:
        for prototype in np.unique(svm_all['Prototype']):
            if 'flag' in prototype or 'eigengene' in prototype:
                for cd in np.unique(svm_all['Center Dimension']):
                    if 'flag' in prototype:
                        for dd in np.unique(svm_all['Data Dimension']):
                            idx = (svm_all['Data Set'] == ds) &\
                                (svm_all['Prototype'] == prototype) &\
                                (svm_all['Center Dimension'] == cd) &\
                                (svm_all['Data Dimension'] == dd)     
                            mean_bsr = np.mean(svm_all[idx]['Relative Gain'])
                            std_bsr = np.std(svm_all[idx]['Relative Gain'])
                            row = pd.DataFrame(data = [[mean_bsr, std_bsr, cd, dd, prototype, ds]], 
                                            columns = mean_svm_data.columns)
                            mean_svm_data = pd.concat([mean_svm_data, row])
                            row = pd.DataFrame(data = [[ds, f'{prototype}_{cd}_{dd}', mean_bsr]], 
                                            columns = simple_mean_svm_data.columns)
                            simple_mean_svm_data = pd.concat([simple_mean_svm_data, row])
                    else:
                        idx = (svm_all['Data Set'] == ds) &\
                                (svm_all['Prototype'] == prototype) &\
                                (svm_all['Center Dimension'] == cd)
                        mean_bsr = np.mean(svm_all[idx]['Relative Gain'])
                        std_bsr = np.std(svm_all[idx]['Relative Gain'])
                        row = pd.DataFrame(data = [[mean_bsr, std_bsr, cd, 1, prototype, ds]], 
                                            columns = mean_svm_data.columns)
                        mean_svm_data = pd.concat([mean_svm_data, row])
                        row = pd.DataFrame(data = [[ds, f'{prototype}_{cd}', mean_bsr]], 
                                            columns = simple_mean_svm_data.columns)
                        simple_mean_svm_data = pd.concat([simple_mean_svm_data, row])
            else:
                idx = (svm_all['Data Set'] == ds) &\
                    (svm_all['Prototype'] == prototype)   
                mean_bsr = np.mean(svm_all[idx]['Relative Gain'])
                std_bsr = np.std(svm_all[idx]['Relative Gain'])
                row = pd.DataFrame(data = [[mean_bsr, std_bsr, 1, 1, prototype, ds]], 
                                    columns = mean_svm_data.columns)
                mean_svm_data = pd.concat([mean_svm_data, row])
                row = pd.DataFrame(data = [[ds, prototype, mean_bsr]], 
                                        columns = simple_mean_svm_data.columns)
                simple_mean_svm_data = pd.concat([simple_mean_svm_data, row])

    simple_mean_svm_data.to_csv(f'{results_dir}/refined_mean_svm_score.csv')
