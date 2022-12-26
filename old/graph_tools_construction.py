import numpy as np
import scipy.cluster.hierarchy as sch
import networkx as nx
from matplotlib import pyplot as plt
import pylab
from scipy.spatial.distance import squareform
import pandas as pd


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


