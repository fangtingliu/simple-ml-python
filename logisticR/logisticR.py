import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import utils;


def loadtxt(X):
    return utils.loadtxt(X)

# two features only
def plot(X, Y):
    m = Y.shape[0]
    pos1 = np.where(Y == 1)
    pos0 = np.where(Y == 0)
    plt.scatter(X[pos1, 0], X[pos1, 1], marker='+', c='blue')
    plt.scatter(X[pos0, 0], X[pos0, 1], marker='o', c='red')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(['Admitted', 'Not Admitted'])
    plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z));

def costFunction(X, Y, **option):
    m = Y.shape[0]
    n = X.shape[1]

    theta = option.get('theta')
    if not theta:
        theta = np.zeros((n + 1, 1))
    XwithOnes = np.hstack((np.ones((m, 1)), X))
    h = sigmoid(np.dot(XwithOnes, theta))

    cost = (
      - np.dot(np.transpose(Y), np.log(h))
      - np.dot(np.transpose(1 - Y), np.log(1 - h))
    ) / m

    grad = np.dot(np.transpose(XwithOnes), (h - Y)) / m;
    return (np.sum(cost), grad)