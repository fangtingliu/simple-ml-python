import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize

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

def cost(theta, X, Y):
    m = Y.shape[0]
    n = X.shape[1]

    if theta is None:
        theta = np.zeros((n + 1, 1))
    else:
        theta = np.reshape(theta, (n + 1, 1))
    XwithOnes = np.hstack((np.ones((m, 1)), X))
    h = sigmoid(np.dot(XwithOnes, theta))

    cost = (
      - np.dot(np.transpose(Y), np.log(h))
      - np.dot(np.transpose(1 - Y), np.log(1 - h))
    ) / m

    return np.sum(cost)

def grad(theta, X, Y):
    m = Y.shape[0]
    n = X.shape[1]

    if theta is None:
        theta = np.zeros((n + 1, 1))
    else:
        theta = np.reshape(theta, (n + 1, 1))

    XwithOnes = np.hstack((np.ones((m, 1)), X))
    h = sigmoid(np.dot(XwithOnes, theta))
    grad = np.dot(np.transpose(XwithOnes), (h - Y)) / m
    return np.ndarray.flatten(grad);

def optimize_bfgs(X, Y, **option):
    n = X.shape[1]
    theta = np.zeros((n + 1, 1))
    [theta_res, cost_min, grad_min] = optimize.fmin_bfgs(cost, theta, grad, args=(X, Y), disp=True)
    plot_arg = option.get('plot')
    if plot_arg:
        plot(X, Y)
        plt.plot()

    return (theta_res, cost_min, grad_min)