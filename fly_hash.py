import os
import struct
import numpy as np
from past.builtins import xrange

SET_MEAN = 100

def load_mnist():
    with open(r'dataset\train-images-idx3-ubyte\train-images.idx3-ubyte', 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype = np.uint8,
                             count = 10000*784).reshape(10000, 784)
    images = images.astype(np.float64)
    return images

def preprocess(data):
    # return data - np.mean(data)
    """ Performs several standardizations on the data.
            1) Makes sure all values are non-negative.
            2) Sets the mean of example to SET_MEAN.
            3) Applies normalization if desired.
    """
    dimension = data.shape[1]
    n = data.shape[0]
    # 1. Add the most negative number per column (ORN) to make all values >= 0.
    for col in xrange(dimension):
        data[:,col] += abs(min(data[:,col]))

    # 2. Set the mean of each row (odor) to be SET_MEAN.
    for row in xrange(n):
        
        # Multiply by: SET_MEAN / current mean. Keeps proportions the same.
        data[row,:] = data[row,:] * ((SET_MEAN / np.mean(data[row,:])))
        # data[row,:] = map(int, data[row,:]) 
        
        assert abs(np.mean(data[row,:]) - SET_MEAN) <= 1

    # 3. Applies normalization.
    data = data.astype(np.float64)        
    data = normalize(data)

    # Make sure all values (firing rates) are >= 0.
    for row in xrange(n):
        for col in xrange(dimension):
            assert data[row,col] >= 0

    return data

def normalize(data):
    return data
    # n = data.shape[0]
    # for row in xrange(n):
    #     data[row,:] = data[row,:]/np.linalg.norm(data[row,:])
    # return data

def createFlyMatrix(kc_num, dimension, p):
    fly_matrix = np.zeros(shape = (kc_num, dimension))

    for row in xrange(kc_num):
        rand_sample = np.random.choice(dimension, int(dimension*p), False)
        fly_matrix[row,:][rand_sample] = 1
        assert sum(fly_matrix[row,:]) == int(dimension*p)

    return fly_matrix

def createGaussianMatrix(hash_length, d, mu = 0, sigma = 1):
    gaussian = np.random.normal(mu, sigma, (hash_length, d))
    return gaussian

def KNN(data, queries, knn_size):
    """
    find k query's nearest neighbor in data, return their index in data
    data: n * d
    queries: m * d, m is the size of queries
    
    dists: m * n
    return:
        knn: m * knn_size
    """
    n, m = data.shape[0], queries.shape[0]
    dists = np.matrix(np.sum(np.square(queries), 
                            axis=1)).T + np.sum(np.square(data), axis=1)
    dists = dists + (-2) * np.dot(queries, np.matrix(data).T)
    dists = np.array(dists)
    
    knn = np.zeros((m, knn_size))
    for i in xrange(m):
        knn[i] = np.argsort(dists[i])[:knn_size]
    
    return knn

def AP(ground_truth, predicted):
    truth, total, precision = 0, 0, 0

    for i in predicted:

        total += 1

        if i in ground_truth:
            truth += 1
            precision += truth / total
            
    return precision / predicted.shape[0]
#     return truth / total

def getHashMatrix(raw_hash_matrix, hash_length, WTA):
    n, kc_num = raw_hash_matrix.shape[0], raw_hash_matrix.shape[1]

    if WTA == 'random':
        rand_indices = np.random.choice(kc_num, hash_length, False)

    hash_matrix = np.zeros((n, kc_num))

    for i in xrange(n):
        # Take all neurons.
        if   WTA == 'all':
            assert hash_length == kc_num
            indices = np.arange(kc_num)

        # Highest firing neurons.
        elif WTA == 'top' or WTA == 'binary':      
            indices = np.argpartition(-raw_hash_matrix[i,:],
                                        hash_length)[:hash_length] 

        # Random neurons. 
        elif WTA == 'random': 
            indices = rand_indices

        else: assert False

        if WTA == 'binary':
            hash_matrix[i][indices] = 1
        else:
            hash_matrix[i][indices] = raw_hash_matrix[i][indices]

    return hash_matrix

def mAP(data, queries, proj_matrix, hash_length, WTA = 'all', knn_size = 200):
    """
    calculate the mAP
    data: n * d
    queries: m * 1, m index in data
    matrix: kc_num * d
    knn_size: the k of k nearest neighbors
    """
    n = data.shape[0]
    d = data.shape[1]
    kc_num = proj_matrix.shape[0]

    # 0. get a raw hash matrix, hash_matrix: n * kc_num
    raw_hash_matrix = np.dot(data, np.transpose(proj_matrix))

    # 1. quantization the raw hash matrix
    raw_hash_matrix = np.floor(raw_hash_matrix/10) 

    # 2. apply WTA to raw hash matrix
    hash_matrix = getHashMatrix(raw_hash_matrix, hash_length, WTA)
    
    # 3. find KNN of queries, knn_origin: m * knn_size
    knn_origin = KNN(data, data[queries], knn_size)
    
    # 4. find KNN of queries' hash, knn_hash: m * knn_size
    knn_hash = KNN(hash_matrix, hash_matrix[queries], knn_size)
    
    # 5. calculate sum of AP
    sum_precision = 0
    for i in xrange(queries.shape[0]):
        sum_precision += AP(knn_origin[i], knn_hash[i])
    
    # 6: divide the AP sum with queries size to get mAP
    return sum_precision / queries.shape[0]

def tuningP():
    """
    find the best p, the probability of 1 in fly hash matrix,
    that maximize the mAP
    """
    data = preprocess(load_mnist())
    n, d = data.shape[0], data.shape[1]
    query_size = 1000
    k = 40
    queries = np.random.choice(n, query_size, False)
    candidates = np.arange(0, 1, 0.1)
    maxP, maxmAP = 0, 0
    for p in candidates:
        tmp = mAP(data, queries, createFlyMatrix(k, d, p), k)
        if tmp > maxmAP:
            maxP, maxmAP = p, tmp
    
    candidates = np.arange(0, maxP, 0.01)
    for p in candidates:
        tmp = mAP(data, queries, createFlyMatrix(k, d, p), k)
        if tmp > maxmAP:
            maxP, maxmAP = p, tmp
    return maxP, maxmAP
