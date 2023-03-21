import numpy as np
from matplotlib import pyplot as plt

from sklearn.manifold import MDS
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

import pandas as pd

import ModuleRefinement.center_algorithms as ca


'''
A script for visualizing synthetic datasets with different subspace dimensions

'''

def calc_distance(X: np.array, Y: np.array, dist_type: str = 'euclidean') -> float:

    if dist_type in ['sine', 'sinesq', 'cosine']:
        dist = ca.calc_error_1_2([X], Y, dist_type)
    elif dist_type == 'min_sinesq':
        sing_vals = np.linalg.svd(X.T @ Y)[1]
        dist = 1-np.min(sing_vals)**2
    elif dist_type in ['wt_euclidean', 'euclidean']:
        dist = np.linalg.norm(X-Y)
    elif dist_type  == 'pearson_dist':
        dist = 1-.5*(1+np.corrcoef(X.flatten(),Y.flatten())[0,1])
    else:
        print(f'dist_type: {dist_type} not recognized')

    return dist

def unitize(row: np.array) -> list:
    #make unit vectors
    row_norm = np.linalg.norm(row)
    if row_norm != 0:
        normalized_point = row / row_norm
    else:
        print('zero column in module data. this column was removed.')
    return normalized_point

def process_data(X: np.array, center_method: str, data_dimension: int = 1) -> list:

    normalized_data = []

    if center_method == 'flag_median' or center_method == 'flag_mean':
        if data_dimension > 1:
            #make knn subspace reps
            neigh = NearestNeighbors(n_neighbors = data_dimension, metric = 'cosine')
            neigh.fit(X.T)
            nns = neigh.kneighbors(X.T, return_distance=False)    
            for nn in nns:
                subspace = np.linalg.qr(X[:,nn])[0][:,:data_dimension]
                normalized_data.append(subspace)
        else:
            #make unit vectors
            for row in X.T:
                normalized_data.append(unitize(np.expand_dims(row, axis = 1)))
    elif center_method == 'eigengene':
        #mean center the columns
        p = X.shape[0]
        column_means = np.repeat(np.expand_dims(np.mean(X, axis = 0), axis = 1).T, p, axis = 0)
        X = X - column_means
        normalized_data = [np.expand_dims(d, axis = 1) for d in X.T]
    else:
        #no normalization
        normalized_data = [np.expand_dims(d, axis = 1) for d in X.T]

    return normalized_data

def distance_matrix(X: list, distance: str, center_method: str) -> np.array:
    '''
    Calculate a chordal distance matrix for the dataset

    Inputs:
        X- list of numpy arrays for the datset
        C- list of numpy arrays for the elements of the codebook
    Outputs:
        Distances- a numpy array with 
            rows corresponding to elements of the codebook and 
            columns corresponding to data
    '''
    n = len(X)
    Distances = np.zeros((n,n))

    distance_conversion = {'l2 correlation': 'cosine',
                            'chordal': 'sine',
                            'max correlation': 'min_sinesq',
                            'correlation': 'sinesq',
                            'euclidean': 'euclidean',
                            'weighted euclidean': 'wt_euclidean',
                            'pearson cor dist': 'pearson_dist'}

    sin_cos = distance_conversion[distance]


    for i in range(n):
        point_i = unitize(X[i])
        for j in range(i+1,n,1):
            if 'flag' not in center_method:
                point_j =  unitize(X[j])
            else:
                point_j = X[j]

            Distances[i,j] = calc_distance(point_i, point_j, sin_cos)
            Distances[j,i] = Distances[i,j].copy()
            
    return Distances


if __name__ == '__main__':

    costs = pd.DataFrame(columns = 
                         ['N Samples', 'Center Dimension', 
                          'Data Dimension', 'Representative', 
                          'Cost'])

    for n_samples in [10,20,30,50,100,140]:
        X, _ = make_blobs(n_samples=n_samples, 
                          centers=1, 
                          n_features=50, 
                          random_state=0)
        # data_egene = process_data(X, 'eigengene')
        data = process_data(X, 'module_expression')
        
        for data_dim in [1,2]:

            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            center_dims = [1,2,4,8]
            for ii in range(4):
                center_dim = center_dims[ii]

                data_subspace = process_data(X, 'flag_mean', data_dim)

                fl_mean = ca.flag_mean(data_subspace, r = center_dim)
                fl_median = ca.irls_flag(data_subspace, r = center_dim, sin_cos = 'sine', n_its = 10)[0]

                if data_dim == 1 and center_dim == 1:
                    
                    eigengene, evr = ca.eigengene(data, r = center_dim, evr = True)
                    print(f'n_samples = {n_samples}, data_dim = 1, center_dim = 1.')
                    print(f'Eigengene EVR = {round(evr, 2)}')
                    # eigengene = unitize(eigengene)
                    module_expression = ca.module_expression(data, centrality = 'degree')
                    module_expression = unitize(module_expression)


                    d_mat = distance_matrix(data_subspace + [fl_mean] + [fl_median] + [eigengene] + [module_expression],
                                            'correlation', 
                                            'flag_mean')

                    d_row = [[n_samples, center_dim, data_dim, 'Flag Mean', np.sum(d_mat[:-4,-4])]]
                    row = pd.DataFrame(columns = costs.columns, data = d_row)
                    costs = pd.concat([costs, row])

                    d_row = [[n_samples, center_dim, data_dim, 'Flag Median', np.sum(d_mat[:-4,-3])]]
                    row = pd.DataFrame(columns = costs.columns, data = d_row)
                    costs = pd.concat([costs, row])

                    d_row = [[n_samples, center_dim, data_dim, 'Eigengene', np.sum(d_mat[:-4,-2])]]
                    row = pd.DataFrame(columns = costs.columns, data = d_row)
                    costs = pd.concat([costs, row])

                    d_row = [[n_samples, center_dim, data_dim, 'Module Expression', np.sum(d_mat[:-4,-2])]]
                    row = pd.DataFrame(columns = costs.columns, data = d_row)
                    costs = pd.concat([costs, row])


                    mds = MDS(n_components = 2, dissimilarity='precomputed')
                    embedded = mds.fit_transform(d_mat)

                    axs[ii].scatter(embedded[:-4,0], embedded[:-4,1], color = 'k', marker = 'x', label = 'Genes')
                    axs[ii].scatter(embedded[-4,0], embedded[-4,1], marker = 's', label = 'Flag Mean')
                    axs[ii].scatter(embedded[-3,0], embedded[-3,1], label = 'Flag Median')
                    axs[ii].scatter(embedded[-2,0], embedded[-2,1], marker = '<', label = 'Eigengene')
                    axs[ii].scatter(embedded[-1,0], embedded[-1,1], marker = '^', label = 'Module Expression')
                    axs[ii].title.set_text(f'Center Dimension {center_dim}')
                    axs[ii].set_xlabel('MDS 1')
                    if ii == 0:
                        axs[ii].set_ylabel('MDS 2')
                        axs[ii].legend(loc = 'lower left')

                elif data_dim == 1:
                    eigengene, evr = ca.eigengene(data, r = center_dim, evr = True)
                    print(f'n_samples = {n_samples}, data_dim = 1, center_dim = {center_dim}.')
                    print(f'Eigengene EVR = {round(evr, 2)}')

                    # eigengene = unitize(eigengene)


                    d_mat = distance_matrix(data_subspace + [fl_mean] + [fl_median] + [eigengene] ,
                                            'correlation', 
                                            'flag_mean')
                    
                    d_row = [[n_samples, center_dim, data_dim, 'Flag Mean', np.sum(d_mat[:-3,-3])]]
                    row = pd.DataFrame(columns = costs.columns, data = d_row)
                    costs = pd.concat([costs, row])

                    d_row = [[n_samples, center_dim, data_dim, 'Flag Median', np.sum(d_mat[:-3,-2])]]
                    row = pd.DataFrame(columns = costs.columns, data = d_row)
                    costs = pd.concat([costs, row])

                    d_row = [[n_samples, center_dim, data_dim, 'Eigengene', np.sum(d_mat[:-3,-1])]]
                    row = pd.DataFrame(columns = costs.columns, data = d_row)
                    costs = pd.concat([costs, row])

                    mds = MDS(n_components = 2, dissimilarity='precomputed')
                    embedded = mds.fit_transform(d_mat)
                    axs[ii].scatter(embedded[:-3,0], embedded[:-3,1], color = 'k', marker = 'x', label = 'Genes')
                    axs[ii].scatter(embedded[-3,0], embedded[-3,1], marker = 's', label = 'Flag Mean')
                    axs[ii].scatter(embedded[-4,0], embedded[-2,1], label = 'Flag Median')
                    axs[ii].scatter(embedded[-1,0], embedded[-1,1], marker = '<', label = 'Eigengene')
                    axs[ii].title.set_text(f'Center Dimension {center_dim}')
                    axs[ii].set_xlabel('MDS 1')
                    if ii == 0:
                        axs[ii].set_ylabel('MDS 2')
                        axs[ii].legend(loc = 'lower left')
                
                else:
                    if center_dim  == 1:
                        d_mat = distance_matrix(data_subspace + [fl_mean] + [fl_median], 'correlation', 'flag_mean')
                    else:
                        d_mat = distance_matrix(data_subspace + [fl_mean] + [fl_median], 'max correlation', 'flag_mean')
                    
                    d_row = [[n_samples, center_dim, data_dim, 'Flag Mean', np.sum(d_mat[:-2,-2])]]
                    row = pd.DataFrame(columns = costs.columns, data = d_row)
                    costs = pd.concat([costs, row])

                    d_row = [[n_samples, center_dim, data_dim, 'Flag Median', np.sum(d_mat[:-2,-1])]]
                    row = pd.DataFrame(columns = costs.columns, data = d_row)
                    costs = pd.concat([costs, row])                 
                    
                    mds = MDS(n_components = 2, dissimilarity='precomputed')
                    embedded = mds.fit_transform(d_mat)
                    axs[ii].scatter(embedded[:-2,0], embedded[:-2,1], color = 'k', marker = 'x', label = 'Genes')
                    axs[ii].scatter(embedded[-2,0], embedded[-2,1], marker = 's', label = 'Flag Mean')
                    axs[ii].scatter(embedded[-1,0], embedded[-1,1], label = 'Flag Median')
                    axs[ii].title.set_text(f'Center Dimension {center_dim}')
                    axs[ii].set_xlabel('MDS 1')
                    if ii == 0:
                        axs[ii].set_ylabel('MDS 2')
                        axs[ii].legend(loc = 'lower left')

            plt.tight_layout()
            plt.savefig(f'experiments/plots/synthetic/mds_{n_samples}_{data_dim}.png')
            plt.close()

    costs.to_csv('./experiments/compare_prototypes/costs.csv')



            









