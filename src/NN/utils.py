import random as rd
import numpy as np
import matplotlib.pyplot as plt


# This function is used to draw a batch of random samples from the training data
def get_batch(X, Y, batch_size):
    """
    X: the training data
    Y: the expectations corresponding to the training data
    batch_size: the size of the batch we need to draw
    """

    x_batch = None
    y_batch = None
    already_drawn_indices = []

    for j in range(batch_size):
        # the index of the random sample is drawn, but we have to make sure that the sample is not already part of
        # the batch we are putting together
        i = rd.randrange(0, X.shape[1])
        while i in already_drawn_indices:
            i = rd.randrange(0, X.shape[1])
        already_drawn_indices.append(i)

        # the sample is added to the batch
        if j == 0:
            x_batch = np.reshape(X[:, i:i + 1], (X.shape[0], 1))
            y_batch = np.reshape(Y[:, i:i + 1], (1, 1))
        else:
            x_batch = np.reshape(np.concatenate((x_batch, X[:, i:i + 1]), axis=1), (X.shape[0], x_batch.shape[1] + 1))
            y_batch = np.reshape(np.concatenate((y_batch, Y[:, i:i + 1]), axis=1), (1, y_batch.shape[1] + 1))

    return x_batch, y_batch


# This function is used as an utility to display the history of the training and validation errors
def display_history(training_error_history, validation_error_history):
    assert len(training_error_history) == len(validation_error_history)

    x_axis = range(len(training_error_history))
    plt.plot(x_axis, training_error_history, label="training", color="red")
    plt.plot(x_axis, validation_error_history, label="validation", color="blue")
    plt.legend()
    plt.show()