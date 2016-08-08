import numpy as np


from utils import utils;

def kMeansInitCentroids(X, K):
    np.random.shuffle(X)
    return X[0:K]

def findClosestCentroid(X, centroids):
    rows = X.shape[0]
    K = centroids.shape[0]
    X_idx = np.zeros(rows)

    for i in range(rows):
        cost = float("inf")
        for k in range(K):
            dis = sum((X[i, :] - centroids[k, :]) ** 2)
            if dis < cost:
                cost = dis
                X_idx[i] = k

    return X_idx


def kMeansCentroids(data, idx, K):
    cols = data.shape[1]
    rows = data.shape[0]
    centroids = np.zeros((K, cols))
    for i in range(K):
        curr = (idx == i).reshape((rows, 1))
        centroids[i] = sum(data * curr) / sum(curr)
    return centroids



def kMeans(data, K, n=1):
    [data_norm, mu, sigma] = utils.normalizeFeature(data)
    init_centroinds = kMeansInitCentroids(data_norm, K)
    print('init_centroinds,', init_centroinds)
    data_idx = findClosestCentroid(data_norm, init_centroinds)
    centroids = kMeansCentroids(data_norm, data_idx, K)
    print(centroids)
