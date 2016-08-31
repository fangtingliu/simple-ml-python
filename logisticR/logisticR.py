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
    X0 = np.reshape(X[:, 0], (m, 1))
    X1 = np.reshape(X[:, 1], (m, 1))
    fig, ax = plt.subplots()
    ax.scatter(X0[Y==1], X1[Y==1], marker='+', color='blue')
    ax.scatter(X0[Y==0], X1[Y==0], marker='o', color='red')
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