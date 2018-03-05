import os
import struct
import numpy as np
from past.builtins import xrange

def load_mnist():
    with open(r'dataset\train-images-idx3-ubyte\train-images.idx3-ubyte', 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype = np.uint8,
                             count = 10000*784).reshape(10000, 784)

    return images

def preprocess(data):
    return data - np.mean(data)


def createFlyMatrix(k, d, p):
    fly_matrix = np.zeros(shape = (k, d))
    for i in range(k):
        for j in range(d):
            if np.random.random() <= p:
                fly_matrix[i][j] = 1
    
    return fly_matrix

def createGaussianMatrix(k, d, mu = 0, sigma = 1):
    gaussian = np.random.normal(mu, sigma, (k, d))
    return gaussian

def createHashMatrix(hash_matrix, WTA, k):
    n, expansion = hash_matrix.shape[0], hash_matrix.shape[1]
    if WTA is True:
        tmp = np.zeros((n, expansion))
        for i in xrange(n):
            indices = np.argsort(-abs(hash_matrix[i]))[:k]
            for j in indices:
                tmp[i][j] = hash_matrix[i][j]
    else:
        tmp = np.zeros((n, k))
        rand = np.random.choice(expansion, k, False)
        for i in xrange(n):
            tmp[i] = hash_matrix[i][rand]
        
    return tmp

def createHashMatrixBinary(hash_matrix, WTA, k):
    n, expansion = hash_matrix.shape[0], hash_matrix.shape[1]
    if WTA is True:
        tmp = np.zeros((n, expansion))
        for i in xrange(n):
            indices = np.argsort(-abs(hash_matrix[i]))[:k]
            tmp[i][indices] = 1
    else:
        tmp = np.zeros((n, k))
        rand = np.random.choice(expansion, k, False)
        for i in xrange(n):
            tmp[i] = hash_matrix[i][rand]
        
    return tmp

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
    dists = np.matrix(np.sum(np.square(queries), axis=1)).T + np.sum(np.square(data), axis=1)
    dists = dists + (-2) * np.dot(queries, np.matrix(data).T)
    dists = np.array(dists)
    
    knn = np.zeros((m, knn_size))
    for i in xrange(m):
        knn[i] = np.argsort(dists[i])[:knn_size]
    
    return knn

def AP(ground_truth, predicted):
    truth, total = 0, 0
    precision = 0
    for i in predicted:
        total += 1
        if i in ground_truth:
            truth += 1
            precision += truth / total
            
    return precision / predicted.shape[0]
#     return truth / total

def mAP(data, queries, matrix, expansion = False, k = -1, WTA = False, 
            binary = False, knn_size = 200):
    """
    calculate the mAP
    data: n * d
    queries: m * 1, m index in data
    matrix: k * d
    knn_size: the k of k nearest neighbors

    expansion: if hash matrix is expanded
    WTA: True means using WTA
    binary: True means hash_matrix should be binary
    """
    # 0. get a hash matrix, hash_matrix: n * k
    hash_matrix = np.array(np.dot(data, np.matrix(matrix).T))
    if expansion is True:
        # now the hash_matrix is n * 20k
        if binary is False:
            hash_matrix = createHashMatrix(hash_matrix, WTA, k)
        else:
            hash_matrix = createHashMatrixBinary(hash_matrix, WTA, k)
    
    # 1. find KNN of queries, knn_origin: m * knn_size
    knn_origin = KNN(data, data[queries], knn_size)
    
    # 2. find KNN of queries' hash, knn_hash: m * knn_size
    knn_hash = KNN(hash_matrix, hash_matrix[queries], knn_size)
    
    # 3. calculate mAP
    sum_precision = 0
    for i in xrange(queries.shape[0]):
        sum_precision += AP(knn_origin[i], knn_hash[i])
    
    # 4: divide the AP sum with queries size
    return sum_precision / queries.shape[0]

def tuningP():
    """
    find the best p, the probability of 1 in fly hash matrix,
    that maximize the mAP
    """
    data = preprocess(load_mnist())
    d = data.shape[1]
    query_size = 1000
    k = 40
    queries = np.random.choice(n, query_size, False)
    candidates = np.arange(0, 1, 0.1)
    maxP, maxmAP = 0, 0
    for p in candidates:
        tmp = mAP(data, queries, createFlyMatrix(k, d, p))
        if tmp > maxmAP:
            maxP, maxmAP = p, tmp
    
    candidates = np.arange(0, maxP, 0.01)
    for p in candidates:
        tmp = mAP(data, queries, createFlyMatrix(k, d, p))
        if tmp > maxmAP:
            maxP, maxmAP = p, tmp
    return maxP
