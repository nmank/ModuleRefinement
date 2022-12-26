'''
This file contains algorithms for calculating module representatives. 
These module representatives are the 
    flag mean
    flag median
    eigengene
    module expression vector

by Nathan Mankovich
'''
import numpy as np

def calc_error_1_2(data: list, Y: np.array, sin_cos: str, labels: list = None) -> float:
    '''
    Calculate objective function value. 

    Inputs:
        data - a list of numpy arrays representing points in Gr(k_i,n)
        Y - a numpy array representing a point on Gr(r,n) 
        sin_cos - a string defining the objective function
                    'cosine' = Maximum Cosine
                    'sine' = Sine Median
                    'sinsq' = Flag Mean
                    'geodesic' = Geodesic Median (CAUTION: only for k_i = r = 1)
                    'l2_med' = geodesic distance if k_i or r > 1
                    'zobs' = a subspace version of zobs
        labels - labels for the features within the data
    Outputs:
        err - objective function value
    '''
    k = Y.shape[1]
    err = 0
    if sin_cos == 'sine':
        for x in data:
            r = np.min([k,x.shape[1]])
            sin_sq = r - np.trace(Y.T @ x @ x.T @ Y)
            #fixes numerical errors
            if sin_sq < 0:
                sin_sq = 0
            err += np.sqrt(sin_sq)
    elif sin_cos == 'cosine':
        for x in data:
            err += np.sqrt(np.trace(Y.T @ x @ x.T @ Y))
    elif sin_cos == 'sinesq':
        for x in data:
            r = np.min([k,x.shape[1]])
            sin_sq = r - np.trace(Y.T @ x @ x.T @ Y)
            #fixes numerical errors
            if sin_sq < 0:
                sin_sq = 0
            err += sin_sq
    elif sin_cos == 'geodesic':
        for x in data:
            cos = (Y.T @ x @ x.T @ Y)[0][0]
            #fixes numerical errors
            if cos > 1:
                cos = 1
            elif cos < 0:
                cos = 0
            geodesic_distance = np.arccos(np.sqrt(cos))
            err += geodesic_distance
    elif sin_cos == 'l2_med':
        for x in data:
            err += gr_dist(x, Y)

    elif sin_cos == 'zobs':

        idx_class0 = np.where(labels == 0)
        idx_class1 = np.where(labels == 1)

        Y0 = Y[idx_class0]
        Y1 = Y[idx_class1]

        for x in data:

            x0 = x[idx_class0]
            x1 = x[idx_class1]

            #sloppy divide by 0 fix
            x0_norm = np.trace(x0.T @ x0)
            if x0_norm == 0:
                x0_norm = 1
            x1_norm = np.trace(x1.T @ x1)
            if x1_norm == 0:
                x1_norm = 1
            Y0_norm = np.trace(Y0.T @ Y0)
            if Y0_norm == 0:
                Y0_norm = 1
            Y1_norm = np.trace(Y1.T @ Y1)
            if Y1_norm == 0:
                Y1_norm = 1


            
            r0 = np.sqrt(np.trace(Y0.T @ x0 @ x0.T @ Y0)/(Y0_norm*x0_norm))
            r1 = np.sqrt(np.trace(Y1.T @ x1 @ x1.T @ Y1)/(Y1_norm*x1_norm))

            z_class0 = np.arctanh(r0)
            z_class1 = np.arctanh(r1)

            zobs = (z_class0-z_class1) / np.sqrt( 1/(len(idx_class0[0])-3) + 1/(len(idx_class1[0])-3))

            zobs = np.abs(zobs)

            err += zobs
    return err


def flag_mean(data: list, r: int) -> np.array:
    '''
    Calculate the Flag Mean

    Inputs:
        data - list of numpy arrays representing points on Gr(k_i,n)
        r - integer number of columns in flag mean
    Outputs:
        mean - a numpy array representing the Flag Mean of the data
    ''' 
    X = np.hstack(data)
    
    mean = np.linalg.svd(X, full_matrices = False)[0][:,:r]

    return mean


def eigengene(data: list, r: int, evr = False) -> np.array:

    #mean center
    p = len(data)
    X = np.hstack(data)
    row_means = np.repeat(np.expand_dims(np.mean(X, axis = 1), axis = 1), p, axis = 1)
    X = X - row_means

    #compute eigengene
    [the_eigengenes, sing_vals, _] = np.linalg.svd(X)
    the_eigengene = the_eigengenes[:,:r]

    evals = sing_vals**2


    if evr:
        explained_variance_ratio = np.sum(evals[:r])/np.sum(evals)
        return the_eigengene, explained_variance_ratio
    
    else:
        return the_eigengene


def flag_mean_iteration(data: list, Y0: np.array, weight: float, eps: float = .0000001) -> np.array:
    '''
    Calculates a weighted Flag Mean of data using a weight method for FlagIRLS
    eps = .0000001 for paper examples

    Inputs:
        data - list of numpy arrays representing points on Gr(k_i,n)
        Y0 - a numpy array representing a point on Gr(r,n)
        weight - a string defining the objective function
                    'sine' = flag median
                    'sinsq' = flag mean
        eps - a small perturbation to the weights to avoid dividing by zero
    Outputs:
        Y- the weighted flag mean
    '''
    r = Y0.shape[1]
    
    aX = []
    al = []

    ii=0

    for x in data:
        if weight == 'sine':
            m = np.min([r,x.shape[1]])
            sinsq = m - np.trace(Y0.T @ x @ x.T @ Y0)
            al.append((max(sinsq,eps))**(-1/4))
        elif weight == 'cosine':
            cossq = np.trace(Y0.T @ x @ x.T @ Y0)
            al.append((max(cossq,eps))**(-1/4))
        elif weight == 'geodesic':
            sinsq = 1 - Y0.T @ x @ x.T @ Y0
            cossq = Y0.T @ x @ x.T @ Y0
            al.append((max(sinsq*cossq, eps))**(-1/4))
        else:
            print('sin_cos must be geodesic, sine or cosine')
        aX.append(al[-1]*x)
        ii+= 1

    Y = flag_mean(aX, r)

    return Y


def irls_flag(data: list, r: int, n_its: int, sin_cos: str, opt_err: str = 'geodesic', init: str = 'random', seed: int = 0) -> tuple: 
    '''
    Use FlagIRLS on data to output a representative for a point in Gr(r,n) 
    which solves the input objection function

    Repeats until iterations = n_its or until objective function values of consecutive
    iterates are within 0.0000000001 and are decreasing for every algorithm (except increasing for maximum cosine)

    Inputs:
        data - list of numpy arrays representing points on Gr(k_i,n)
        r - the number of columns in the output
        n_its - number of iterations for the algorithm
        sin_cos - a string defining the objective function for FlagIRLS
                    'sine' = flag median
        opt_err - string for objective function values in err (same options as sin_cos)
        init - string 'random' for random initlalization. 
            otherwise input a numpy array for the inital point
        seed - seed for random initialization, for reproducibility of results
    Outputs:
        Y - a numpy array representing the solution to the chosen optimization algorithm
        err - a list of the objective function values at each iteration (objective function chosen by opt_err)
    '''
    err = []
    n = data[0].shape[0]


    #initialize
    if init == 'random':
        #randomly
        np.random.seed(seed)
        Y_raw = np.random.rand(n,r)-.5
        Y = np.linalg.qr(Y_raw)[0][:,:r]
    elif init == 'data':
        np.random.seed(seed)
        Y = data[np.random.randint(len(data))]
    else:
        Y = init

    err.append(calc_error_1_2(data, Y, opt_err))

    #flag mean iteration function
    #uncomment the commented lines and 
    #comment others to change convergence criteria
    
    itr = 1
    diff = 1
    while itr <= n_its and diff > 0.0000000001: #added abs
        Y0 = Y.copy()
        Y = flag_mean_iteration(data, Y, sin_cos)
        err.append(calc_error_1_2(data, Y, opt_err))
        if opt_err == 'cosine':
            diff  = err[itr] - err[itr-1]
        else:
            diff  = err[itr-1] - err[itr]
        # diff  = np.abs(err[itr-1] - err[itr])
           
        itr+=1
    


    if diff > 0:
        return Y, err
    else:
        return Y0, err[:-1]


def adjacency_matrix(X: np.array, msr: str = 'parcor', epsilon: float = 0, 
                     h_k_param: float = 2, negative: bool  = False, 
                     h_k_ord: int = 2):
    '''
    A function that builds an adjacecny matrix out of data using two methods.

    Nodes are columns of the data matrix X!

    inputs: data matrix 
                a numpy array with n rows m columns (m data points living in R^n)
            msr
                a string for method for calculating distance between data points 
                corrolation or heatkernel or partial correlation
            epsilon
                a number that is a user-parameter that determines will disconnect 
                all points that are further away or less corrolated than epsilon
            h_k_param
                a number for the heat parameter for the heat kernel similarity measure
            weighted
                a boolean that creates a weighted matrix if true
            negative 
                a boolean to include negative correlations? (default is False)
    outputs: adjacency matrix
                    represents a directed weighted graph of the data (dimensions m x m)
    '''
    n,m = X.shape

    if msr == 'correlation':
        norms = np.repeat(np.expand_dims(np.linalg.norm(X, axis=0),axis= 0), n, axis=0)
        norms[np.where(norms==0)] = 1 #so we don't divide by 0s
        normalized_X = X/norms
        AdjacencyMatrix = normalized_X.T @ normalized_X - np.eye(m)
        if not negative:
            AdjacencyMatrix  = np.abs(AdjacencyMatrix)
        AdjacencyMatrix[np.where(AdjacencyMatrix > 1)] = 1


    elif msr == 'heatkernel':
        AdjacencyMatrix = np.zeros((m,m))

        for i in range(m):
            for j in range(i+1,m):
                AdjacencyMatrix[i,j] = np.exp(-(np.linalg.norm( X[:,i]-X[:,j], ord = h_k_ord)**2 )/(2*h_k_param))
                AdjacencyMatrix[j,i] = AdjacencyMatrix[i,j].copy()

    if epsilon > 0:
        AdjacencyMatrix[AdjacencyMatrix < epsilon] = 0

    #force diagonal 0
    np.fill_diagonal(AdjacencyMatrix, 0)

    return AdjacencyMatrix


def centrality_scores(A: np.array, centrality: str = 'degree', pagerank_d: float = .85, 
                      pagerank_seed: int = 1, stochastic: bool = False, 
                      in_rank: bool = False):
    '''
    A method for computing the centrality of the nodes in a network

    Note: a node has degree 5 if it has 5 edges coming out of it. 
    We are interested in out edges rather than in edges! 
    Page rank ranks nodes with out edges higher than nodes with in-edges.
    
    Inputs:
        A - a numpy array that is the adjacency matrix
        centrality - a string for the type of centrality
                     options are:
                         'largest_evec'
                         'page_rank'
                         'degree'
        pagerank_d - float, parameter for pagerank 
    Outputs:
        scores - a numpy array of the centrality scores for the nodes in the network
                 index in scores corresponds to index in A
    
    '''
    
    if centrality == 'large_evec':
        W,V = np.linalg.eig(A)
        scores = np.real(V[:,W.argmax()])

    elif centrality == 'degree':
        #sum by out edges
        degrees = np.sum(A,axis = 0)
        if A.shape[0] > 1:
            scores = degrees
        else:
            scores = np.array([0])
        
    elif centrality == 'page_rank':
        if not in_rank:
            A = A.T
        n = A.shape[0]
        if n == 1:
            scores = np.array([0])
        else:
            #in connections
            connected_idx = np.where(np.sum(A, axis = 0) != 0)[0]
            #connected_idx_out = np.where(np.sum(A, axis = 1) != 0)[0]
            #connected_idx = np.union1d(connected_idx_in, connected_idx_out)
            connected_A = A[:,connected_idx][connected_idx,:]
            n = len(connected_idx)
            if n <= 1:
                scores = np.array([0])
            else:
                M = np.zeros((n,n))
                for i in range(n): 
                    A_sum = np.sum(connected_A[:,i])
                    if A_sum == 0:
                        M[:,i] = connected_A[:,i]
                        # print('dangling nodes for page rank')
                    else:
                        M[:,i] = connected_A[:,i]/A_sum

                if stochastic:
                    #taken from da wikipedia
                    eps = 0.001

                    #new and fast
                    np.random.seed(pagerank_seed)
                    
                    v = np.random.rand(n, 1)
                    v = v / np.linalg.norm(v, 1)
                    err = 1
                    while err > eps:
                        v0 = v.copy()
                        v = (pagerank_d * M) @ v0 + (1 - pagerank_d) / n
                        err = np.linalg.norm(v - v0, 2)

                    #sanity check
                    big_M = (pagerank_d * M)  + np.ones((n,n))*(1 - pagerank_d) / n
                    v_check = big_M @ v
                    if not np.allclose(v_check, v, rtol=1e-05, atol=1e-08):
                        print('page rank not converged')
                else:
                    big_M = (pagerank_d * M)  + np.ones((n,n))*(1 - pagerank_d) / n
                    evals, evecs = np.linalg.eig(big_M)
                    dist_from_1 = np.abs(evals - 1)
                    idx = dist_from_1.argmin()
                    v = evecs[:,idx]
                    v = v/np.sum(v)
                    if np.abs(evals[idx]-1) > 1e-08:
                        print('page rank not converged')
                    
                connected_scores = v.flatten()

                scores = np.zeros(A.shape[0])
                scores[connected_idx] = connected_scores


        
    else:
        print('centrality type not recognized')
        
    return scores

def module_expression(module_data: list, centrality: str = None) -> np.array:
        if centrality is not None:
            A = adjacency_matrix(np.hstack(module_data), msr = 'correlation')
            scores = centrality_scores(A, centrality = centrality)
            max_score = np.max(scores) 
            if max_score != 0:
                scores = scores / max_score

            scored_module = [scores[i]*module_data[i] for i in range(len(module_data))]
        else:
            scored_module = module_data
        center = np.expand_dims(np.mean(np.hstack(scored_module),axis = 1),axis = 1)
        return center
