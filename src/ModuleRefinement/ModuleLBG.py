import numpy as np
from sklearn.base import BaseEstimator
import ModuleRefinement.center_algorithms as ca
from sklearn.neighbors import NearestNeighbors



class ModuleLBG(BaseEstimator):


    def __init__(self, center_method: str = 'eigengene', center_dimension: int = 1, data_dimension: int = 1,
                centers: list = [], distance: str = 'correlation', centrality: str = '',
                epsilon: float = .0001, errs: list = [], distortions: list = [], n_centers: int = 0,
                center_seed: int = 1):
        # set params
        self.center_method_ = str(center_method)
        self.center_dimension_ = int(center_dimension)
        self.data_dimension_ = int(data_dimension)
        self.centers_ = list(centers)
        self.distance_ = str(distance)
        self.centrality_ = str(centrality)
        self.epsilon_ = float(epsilon)
        self.errs_ = list(errs)
        self.distortions_ = list(distortions)
        self.n_centers_ = int(n_centers)
        self.center_seed_ = int(center_seed)

    @property
    def center_method(self):
        return self.center_method_

    @property
    def center_dimension(self):
        return self.center_dimension_

    @property
    def data_dimension(self):
        return self.data_dimension_
  
    @property
    def centers(self):
        return self.centers_

    @property
    def distance(self):
        return self.distance_
    
    @property
    def centrality(self):
        return self.centrality_

    @property
    def epsilon(self):
        return self.epsilon_

    @property
    def errs(self):
        return self.errs_
    
    @property
    def n_centers(self):
        return self.n_centers_

    @property
    def distortions(self):
        return self.distortions_
    
    @property
    def center_seed(self):
        return self.center_seed_
        
    def fit_transform(self, X: np.array, y: np.array = None) -> None:
        if type(X) == np.array:
            data_list = self.process_data(X)
        elif type(X) == list:
            data_list = X
        else:
            print('invalid X type')
        self.lbg(data_list)

    def unitize(self, row: np.array) -> list:
        #make unit vectors
        row_norm = np.linalg.norm(row)
        if row_norm != 0:
            normalized_point = row / row_norm
        else:
            print('zero column in module data. this column was removed.')
        return normalized_point

    def process_data(self, X: np.array) -> list:

        normalized_data = []

        if self.center_method_ == 'flag_median' or self.center_method_ == 'flag_mean':
            if self.data_dimension_ > 1:
                #make knn subspace reps
                neigh = NearestNeighbors(n_neighbors = self.data_dimension_, metric = 'cosine')
                neigh.fit(X.T)
                nns = neigh.kneighbors(X.T, return_distance=False)    
                for nn in nns:
                    subspace = np.linalg.qr(X[:,nn])[0][:,:self.data_dimension_]
                    normalized_data.append(subspace)
            else:
                #make unit vectors
                for row in X.T:
                    normalized_data.append(self.unitize(np.expand_dims(row, axis = 1)))
        elif self.center_method_ == 'eigengene':
            #mean center the columns
            p = X.shape[0]
            column_means = np.repeat(np.expand_dims(np.mean(X, axis = 0), axis = 1).T, p, axis = 0)
            X = X - column_means
            normalized_data = [np.expand_dims(d, axis = 1) for d in X.T]
        else:
            #no normalization
            normalized_data = [np.expand_dims(d, axis = 1) for d in X.T]

        return normalized_data
    
    def calc_distance(self, X: np.array, Y: np.array, dist_type: str = 'euclidean', weight: float = 1) -> float:

        if dist_type in ['sine', 'sinesq', 'cosine']:
            dist = ca.calc_error_1_2([X], Y, dist_type)*weight
        elif dist_type == 'min_sinesq':
            sing_vals = np.linalg.svd(X.T @ Y)[0]
            dist = 1-np.min(sing_vals)**2
        elif dist_type in ['wt_euclidean', 'euclidean']:
            dist = np.linalg.norm(X-Y)*weight
        elif dist_type  == 'pearson_dist':
            dist = 1-.5*(1+np.corrcoef(X.flatten(),Y.flatten())[0,1])
        else:
            print(f'dist_type: {dist_type} not recognized')

        return dist

    def distance_matrix(self, X: list, weights: np.array = None) -> np.array:
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
        m = len(self.centers_)
        Distances = np.zeros((m,n))

        distance_conversion = {'l2 correlation': 'cosine',
                               'chordal': 'sine',
                               'max correlation': 'min_sinesq',
                               'correlation': 'sinesq',
                               'euclidean': 'euclidean',
                               'weighted euclidean': 'wt_euclidean',
                               'pearson cor dist': 'pearson_dist'}

        sin_cos = distance_conversion[self.distance_]

        if weights is None:
            weights = np.ones(n)

        for i in range(m):
            if 'module_expression' in self.center_method_:
                center_i = self.unitize(self.centers_[i])
            else:
                center_i = self.centers_[i]   
            for j in range(n):
                if 'flag' not in self.center_method_:
                    point_j =  self.unitize(X[j])
                else:
                    point_j = X[j]

                Distances[i,j] = self.calc_distance(center_i, point_j, sin_cos, weights[j])
                
        return Distances

    def closest_center(self, d_mat: np.array) -> np.array:
        #find the closest center for each point
        if self.distance_ == 'l2 correlation':
            index  = np.argmax(d_mat, axis = 0)
        else:
            index = np.argmin(d_mat, axis = 0)
        return index

    def random_centers(self, X: list) -> None:
        np.random.seed(self.center_seed_)
        self.centers_ = []
        for _ in range(self.n_centers_):
            self.centers_.append(X[np.random.randint(0,len(X))])

    def calc_centers(self, X: list, index: np.array) -> None:
        m = len(self.centers_)
        self.centers_ = []
        for c in np.unique(index):
            idx = np.where(index == c)[0]
            if len(idx) > 0:
                if self.center_method_ == 'flag_mean':
                    self.centers_.append(ca.flag_mean([X[i] for i in idx], self.center_dimension_))
                elif self.center_method_ == 'eigengene':
                    self.centers_.append(ca.eigengene([X[i] for i in idx], self.center_dimension_))
                elif self.center_method_ == 'flag_median':
                    self.centers_.append(ca.irls_flag([X[i] for i in idx], self.center_dimension_, 10, 'sine', 'sine')[0])
                elif self.center_method == 'module_expression': 
                    self.centers_.append(ca.module_expression([X[i] for i in idx], self.centrality_))
                else:
                    print('center_method not recognized.')

    def get_labels(self, X: list, weights: np.array = None) -> np.array:
        #calculate distance matrix
        d_mat = self.distance_matrix(X, weights)

        #find closest center
        index = self.closest_center(d_mat)
        
        return index

    def cluster_purity(self, X: list, labels_true: list) -> float:
        '''
        Calculate the cluster purity of the dataset

        Inputs:
            X- list of numpy arrays for the dataset
            centers- a list of numpy arrays for the codebook
            labels_true- a list of the true labels
        Outputs:
            purity- a float for the cluster purity
        '''

        index = self.get_labels(X)
        
        count = 0
        for i in range(len(self.centers_)):
            idx = np.where(index == i)[0]
            if len(idx) != 0:
                cluster_labels = [labels_true[i] for i in idx]
                most_common_label = max(set(cluster_labels), key = cluster_labels.count)
                # count += cluster_labels.count(most_common_label)
                count += cluster_labels.count(most_common_label)/len(idx)

        # return count/len(X)
        purity = count/len(self.centers_)
        return purity

    def lbg(self, X: list, weights: np.array = None) -> bool:
        '''
        LBG clustering with module representatives
        '''
        
        # n_pts = len(X)
        error = 1
        self.distortions_ = []

        #init centers if centers aren't provided
        if len(self.centers_) == 0:
            self.random_centers(X)
        else:
            self.n_centers_ = len(self.centers_)

        #calculate distance matrix
        d_mat = self.distance_matrix(X, weights)

        #find the closest center for each point
        index = self.closest_center(d_mat)

        #calculate first distortion
        new_distortion = np.sum(d_mat[index])

        self.distortions_.append(new_distortion)

        max_itrs = 20        

        n_itrs = 1
        self.errs_ = []
        while error > self.epsilon and n_itrs <= max_itrs:
            # print(f'iteration {len(self.errs_)}')

            #set new distortion as old one
            old_distortion = new_distortion

            #calculate new centers
            self.calc_centers(X, index)

            #calculate distance matrix
            d_mat = self.distance_matrix(X, weights)

            #find the closest center for each point
            index = self.closest_center(d_mat)

            #new distortion
            new_distortion = np.sum(d_mat[index])
            
            self.distortions_.append(new_distortion)

            #termination_criteria
            if new_distortion <0.00000000001:
                error = 0
            else:
                error = np.abs(new_distortion - old_distortion)/old_distortion
            self.errs_.append(error)
            
            n_itrs +=1

        if n_itrs == max_itrs:
            print(f'max iterations of {max_itrs} reached!')
        
