import numpy as np
import pandas as pd
import plotly.express as px

from plotly.subplots import make_subplots
import plotly.graph_objects as go

go_all = pd.read_csv('../results/svm_score_new_labels.csv', index_col = 0)

wgcna_svm = pd.read_csv('../results/new_svm_wgcna_score.csv', index_col = 0)
 
center_methods = [['eigengene',1,1], 
                     ['flag_mean',1,1], 
                     ['flag_median',1,1], 
                     ['module_expression',1,1],
                     ['eigengene',2,1], 
                     ['flag_mean',2,1], 
                     ['flag_median',2,1],
                     ['eigengene',4,1], 
                     ['flag_mean',4,1], 
                     ['flag_median',4,1],
                     ['eigengene',8,1], 
                     ['flag_mean',8,1], 
                     ['flag_median',8,1],
                     ['flag_mean',1,2], 
                     ['flag_median',1,2],
                     ['flag_mean',2,2], 
                     ['flag_median',2,2],
                     ['flag_mean',4,2], 
                     ['flag_median',4,2],
                     ['flag_mean',8,2], 
                     ['flag_median',8,2]]
data_sets = np.unique(go_all["Data Set"])

go_better = pd.DataFrame(columns = ['Relative Gain', 'Fold', 'Prototype', 'Data Dimension', 'Center Dimension', 'Data Set'])
for data_set in data_sets:
        for prototype, center_dim, data_dim in center_methods:
                for fold in range(5):
                        wgcna_query = (wgcna_svm["Algorithm"] == 'WGCNA') &\
                                      (wgcna_svm["Fold"] == fold) &\
                                      (wgcna_svm["Data Set"] == data_set)
                        if np.sum(wgcna_query) > 0:
                                wgcna_total = wgcna_svm[wgcna_query]["BSR"].sum()
                                query = (go_all["Algorithm"] == 'WGCNA_LBG') &\
                                        (go_all["Central Prototype"] == prototype) &\
                                        (go_all["Data Dimension"] == data_dim) &\
                                        (go_all["Center Dimension"] == center_dim) &\
                                        (go_all["Fold"] == fold) &\
                                        (go_all["Data Set"] == data_set) &\
                                        (go_all["Size"] <= 22276)
                                if np.sum(query) > 0:
                                        eigengene_total = go_all[query]["BSR"].sum()
                                        # print((mean_feats['wgcna']/mean_feats[prototype]))
                                        relative_gain = (eigengene_total/(wgcna_total+.0001))-1
                                        row = pd.DataFrame(columns = go_better.columns,
                                                        data = [[relative_gain, fold, prototype, data_dim, center_dim, data_set]])
                                        go_better = go_better.append(row, ignore_index = True)

go_better.to_csv('../results/refined_svm_score_new_labels.csv')

svm_all = go_better.copy()

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

# mean_svm_data.to_csv('./results/refined_mean_svm_score.csv')
simple_mean_svm_data.to_csv('../results/refined_simple_mean_svm_score_new_labels.csv')
