import numpy as np

def normalizeFeature(X):
    if len(X[0]) > 1:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean)/std, mean, std
    else:
        return X, None, None