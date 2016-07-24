import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.path.dirname(os.path.realpath(__file__))

def loadtxt(path):
    data = np.loadtxt(path, delimiter=",", dtype="float")
    rlen = len(data)
    clen = len(data[0])
    X = data[:, 0:(clen - 1)]
    Y = data[:, (clen - 1):]
    return (X, Y)

def normalizeFeature(X):
    if len(X[0]) > 1:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return np.divide((X - mean), std), mean, std
    else:
        return X, None, None

def plot(X, Y, markerType, title=None, xlabel=None, ylabel=None):
    plt.plot(X, Y, markerType)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.close()


def cost(thetas, X, Y):
    m = len(X)
    h = np.dot(X, thetas)
    J = np.sum(np.square(h - Y)) / (2 * m)
    return J


def gradientDescent(X, Y, learn_rate, num_iters):
    m = len(Y)
    XwithOnes = np.hstack((np.ones((m, 1)), X))
    thetas = np.zeros((len(XwithOnes[0, :]), 1))
    J_hist = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = np.dot(XwithOnes, thetas)
        thetas = thetas - learn_rate * np.dot(np.transpose(XwithOnes), (h - Y)) / m
        J_hist[i] = cost(thetas, XwithOnes, Y)
    return (thetas, J_hist)

# One feature only
def plotFitLine(X, Y, thetas):
    m = len(Y)
    XwithOnes = np.hstack((np.ones((m, 1)), X))
    h = np.dot(XwithOnes, thetas)
    plt.plot(X, Y, 'b+')
    plt.title("Fitting")
    plt.plot(X, h, 'r-')
    plt.show()
    plt.close()

def costSurfPlot(X, Y, s0=-10, e0=10, s1=-10, e1=10):
    th0 = np.linspace(s0, e0, 100.)
    th1 = np.linspace(s1, e1, 100.)
    m = len(Y)
    J = np.zeros((100, 100))
    X = np.hstack((np.ones((m, 1)), X))
    for i in range(100):
        for j in range(100):
            J[i][j] = cost([th0[i], th1[j]], X, Y)
    fig = plt.figure()
    Th0, Th1 = np.meshgrid(th0, th1)
    threeD = fig.add_subplot(111, projection="3d")
    threeD.plot_surface(Th0, Th1, J)
    threeD.set_xlabel("theta_0")
    threeD.set_ylabel("theta_1")
    threeD.set_zlabel("cost")
    plt.title("Cost on Thetas")
    plt.show()
    plt.close()

def costContourPlot(X, Y, thetas=[None, None], s0=-10, e0=10, s1=-10, e1=10):
    th0 = np.linspace(s0, e0, 200.)
    th1 = np.linspace(s1, e1, 200.)
    m = len(Y)
    J = np.zeros((200, 200))
    X = np.hstack((np.ones((m, 1)), X))
    for i in range(200):
        for j in range(200):
            J[i][j] = cost([th0[i], th1[j]], X, Y)
    Th0, Th1 = np.meshgrid(th0, th1)
    plt.contour(Th0, Th1, J)
    if thetas is not [None, None]:
        plt.plot(thetas[0][0], thetas[1][0], 'rx');
    plt.xlabel("theta_0")
    plt.xticks(np.arange(s0, e0, (e0 - s0) / 10.))
    plt.ylabel("theta_1")
    plt.yticks(np.arange(s1, e1, (e1 - s1) / 10.))
    plt.show()
    plt.close()

# No need to normalize features
def normalEq(X, Y):
    m = len(Y)
    X = np.hstack((np.ones((m, 1)), X))
    Xtr = np.transpose(X)
    return np.dot(np.dot(np.linalg.inv(np.dot(Xtr, X)), Xtr), Y)

def predict(thetas, X):
    return np.dot(np.hstack(([1], X)), thetas)






