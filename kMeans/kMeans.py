from copy import copy

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage

from utils import utils;

def kMeansInitCentroids(X, K):
    toBeShuffledX = copy(X)
    np.random.shuffle(toBeShuffledX)
    return toBeShuffledX[0:K]

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
        total = np.sum(data * curr, axis=0)
        centroids[i] = total / sum(curr)
    return centroids

def kMeans(data, K, iter_num=50, n=1):
    [data_norm, mu, sigma] = utils.normalizeFeature(data)
    init_centroinds = kMeansInitCentroids(data_norm, K)
    for i in range(iter_num):
        data_idx = findClosestCentroid(data_norm, init_centroinds)
        centroids = kMeansCentroids(data_norm, data_idx, K)
    print(centroids * sigma) + mu
    return (centroids * sigma) + mu


def kMeansImage(path, K, iter_num=50, n=1):
    pic = ndimage.imread(path)
    plt.figure()
    plt.subplot(121)
    plt.imshow(pic)

    picR = pic.shape[0]
    picC = pic.shape[1]

    data = copy(pic)
    data = np.reshape(data, (picC * picR, 3))
    pixels = data.shape[0]
    centroids = kMeansInitCentroids(data, K)
    for i in range(iter_num):
        data_idx = findClosestCentroid(data, centroids)
        centroids = kMeansCentroids(data, data_idx, K)
    kdata = np.zeros((pixels, 3))
    kdata_pic = np.empty([picR, picC])

    data_idx = findClosestCentroid(data, centroids)
    for j in range(pixels):
        kdata[j] = centroids[int(data_idx[j])]

    kdata_pic = np.reshape(kdata, (picR, picC, 3))
    plt.subplot(122)
    plt.imshow(kdata_pic)
    plt.show()
    print centroids
