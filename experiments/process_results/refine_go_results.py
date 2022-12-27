import numpy as np
import pandas as pd

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--wgcna_file", default = 'experiments/results/go_wgcna_score.csv', type = str, help = "path to wgcna go results file")
    parser.add_argument("--lbg_file", default = 'experiments/results/go_lbg_score.csv', type = str, help = "path to lbg go results file")
    parser.add_argument("--center_methods_file", default = 'experiments/compute_modules/center_methods.csv', type = str, 
                        help = "path to csv file with 3 columns: 1) center type, 2) center dimension, 3) data dimension")
    parser.add_argument("--results_dir", default = 'experiments/results/', type = str, help = "path to results directory")
    args = parser.parse_args()

    wgcna_file = args.wgcna_file
    lbg_file = args.lbg_file
    center_methods_file = args.center_methods_file
    results_dir = args.results_dir

    wgcna_go = pd.read_csv(wgcna_file, index_col = 0)

    lbg_go = pd.read_csv(lbg_file, index_col = 0)
    
    center_methods = np.array(pd.read_csv(center_methods_file, index_col = 0))

    data_sets = np.unique(lbg_go["Data Set"])

    go_better = pd.DataFrame(columns = ['Relative Gain', 'Fold', 'Prototype', 'Data Dimension', 'Center Dimension', 'Data Set'])
    for data_set in data_sets:
            for prototype, center_dim, data_dim in center_methods:
                    for fold in range(5):
                            wgcna_query = (wgcna_go["Fold"] == str(fold)) &\
                                        (wgcna_go["Data Set"] == data_set)
                            if np.sum(wgcna_query) > 0:
                                    wgcna_total = wgcna_go[wgcna_query]["GO Significance"].sum()
                                    eg_query = (lbg_go["Algorithm"] == 'WGCNA_LBG') &\
                                            (lbg_go["Central Prototype"] == prototype) &\
                                            (lbg_go["Data Dimension"] == data_dim) &\
                                            (lbg_go["Center Dimension"] == center_dim) &\
                                            (lbg_go["Fold"] == str(fold)) &\
                                            (lbg_go["Data Set"] == data_set)                           
                                    eigengene_total = lbg_go[eg_query]["GO Significance"].sum()
                                    if np.sum(eg_query) > 0:
                                            # print((mean_feats['wgcna']/mean_feats[prototype]))
                                            relative_gain = (eigengene_total/(wgcna_total+.0001))-1
                                            row = pd.DataFrame(columns = go_better.columns,
                                                            data = [[relative_gain, fold, prototype, data_dim, center_dim, data_set]])
                                            go_better = go_better.append(row, ignore_index = True)


    go_better.to_csv(f'{results_dir}/refined_go_score.csv')

    lbg_go = go_better.copy()

    data_sets = np.unique(lbg_go['Data Set'])
    data_mask = ['ebola' not in d for d  in data_sets]
    data_sets = data_sets[data_mask]
    idx = [w in data_sets for w in lbg_go['Data Set']]
    lbg_go = lbg_go[idx]


    #make a data frame with mean and standard deviation for each method
    mean_go_data = pd.DataFrame(columns = ['Mean Relative Gain', 'Std Relative Gain', 'Center Dimension', 
                                            'Data Dimension', 'Prototype', 'Data Set'])
    simple_mean_go_data = pd.DataFrame(columns = ['Data Set', 'Module Refinement Method', 'Mean Relative Gain'])
    for ds in data_sets:
        for prototype in np.unique(lbg_go['Prototype']):
            if 'flag' in prototype or 'eigengene' in prototype:
                for cd in np.unique(lbg_go['Center Dimension']):
                    if 'flag' in prototype:
                        for dd in np.unique(lbg_go['Data Dimension']):
                            idx = (lbg_go['Data Set'] == ds) &\
                                (lbg_go['Prototype'] == prototype) &\
                                (lbg_go['Center Dimension'] == cd) &\
                                (lbg_go['Data Dimension'] == dd)     
                            mean_go = np.mean(lbg_go[idx]['Relative Gain'])
                            std_go = np.std(lbg_go[idx]['Relative Gain'])
                            row = pd.DataFrame(data = [[mean_go, std_go, cd, dd, prototype, ds]], 
                                            columns = mean_go_data.columns)
                            mean_go_data = pd.concat([mean_go_data, row])
                            row = pd.DataFrame(data = [[ds, f'{prototype}_{cd}_{dd}', mean_go]], 
                                            columns = simple_mean_go_data.columns)
                            simple_mean_go_data = pd.concat([simple_mean_go_data, row])
                    else:
                        idx = (lbg_go['Data Set'] == ds) &\
                                (lbg_go['Prototype'] == prototype) &\
                                (lbg_go['Center Dimension'] == cd)
                        mean_go = np.mean(lbg_go[idx]['Relative Gain'])
                        std_go = np.std(lbg_go[idx]['Relative Gain'])
                        row = pd.DataFrame(data = [[mean_go, std_go, cd, 1, prototype, ds]], 
                                            columns = mean_go_data.columns)
                        mean_go_data = pd.concat([mean_go_data, row])
                        row = pd.DataFrame(data = [[ds, f'{prototype}_{cd}', mean_go]], 
                                            columns = simple_mean_go_data.columns)
                        simple_mean_go_data = pd.concat([simple_mean_go_data, row])
            else:
                idx = (lbg_go['Data Set'] == ds) &\
                    (lbg_go['Prototype'] == prototype)   
                mean_go = np.mean(lbg_go[idx]['Relative Gain'])
                std_go = np.std(lbg_go[idx]['Relative Gain'])
                row = pd.DataFrame(data = [[mean_go, std_go, 1, 1, prototype, ds]], 
                                    columns = mean_go_data.columns)
                mean_go_data = pd.concat([mean_go_data, row])
                row = pd.DataFrame(data = [[ds, prototype, mean_go]], 
                                        columns = simple_mean_go_data.columns)
                simple_mean_go_data = pd.concat([simple_mean_go_data, row])

    simple_mean_go_data.to_csv(f'{results_dir}/refined_mean_go_score.csv')
