import numpy as np

def normalizeFeature(X):
    if len(X[0]) > 1:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean)/std, mean, std
    else:
        return X, 0, 1

def loadtxt(path):
    data = np.loadtxt(path, delimiter=",", dtype="float")
    rlen = len(data)
    clen = len(data[0])
    X = data[:, 0:(clen - 1)]
    Y = data[:, (clen - 1):]
    return (X, Y)