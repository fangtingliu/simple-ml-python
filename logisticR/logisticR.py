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
    plt.close()