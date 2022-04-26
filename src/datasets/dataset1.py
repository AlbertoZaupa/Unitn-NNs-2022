import numpy as np

DATASET_SIZE = 10000


# this function is used to generate the feature vectors of our dataset
def x():
    """
    the function simply returns a list of vectors with random values
    """
    return np.random.random((2, DATASET_SIZE))


# this function determines the mapping that our neural network should learn
def y(x):
    """
    the mapping corresponds to f(x) = 5(sin(x) + cos(x)), with some added white noise
    """
    y = 5*(np.sin(x[:1, :]) + np.cos(x[1:, :])) + np.random.normal(size=(1, DATASET_SIZE))

    return np.reshape(y, newshape=(1, DATASET_SIZE))


def dataset():
    """
    returns the whole dataset, already divided in training and validations subsets
    """

    X = x()
    np.random.shuffle(X)
    Y = y(X)
    offset = int(DATASET_SIZE*0.8)

    X_train = X[:, :offset]
    X_val = X[:, offset:]
    Y_train = Y[:, :offset]
    Y_val = Y[:, offset:]

    return X_train, Y_train, X_val, Y_val

