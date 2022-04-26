import numpy as np
from .utils import get_batch


# This function is a simple implementation of mini batch gradient descent.
# It is not intended to be computationally efficient, it is just a demonstration of how one could go about
# implementing mini batch GD
def mini_batch_gradient_descent(X_train, Y_train, X_val, Y_val, neural_network, batch_size, n_iterations):
    """
    X_train: the matrix of training data
    Y_train: the vector of expectations associated with the training data
    X_val: the matrix of validation data
    Y_val: the vector of expectations associated with the validation data
    neural_network: the neural network to be trained
    batch_size: the desired batch size
    n_iterations: the number of iterations of the algorithm that should be performed
    The function returns the lists of values taken by training and validation error at each training step.
    """

    training_error_history = []
    validation_error_history = []
    optimal_weights = None
    min_validation_error = -1

    # the function used to compute the learning rate. For the first iteration nabla is equal to 0.1/batch_size
    def nabla(t):
        return 0.1/((t+1)*batch_size)

    # the training iterations are
    for t in range(n_iterations):
        # a batch of random samples is drawn from the training data
        x_batch, y_batch = get_batch(X_train, Y_train, batch_size)

        # the training iteration is performed
        training_error_history.append(neural_network.train(x_batch, y_batch, nabla(t)))

        # the validation error is computed
        _, validation_error = neural_network.predict(X_val, Y_val)
        validation_error_history.append(validation_error)

        # if the validation error for this iteration is the lowest seen so far, the current weights of the network
        # are saved as the optimal weights
        if t == 0:
            min_validation_error = validation_error
            optimal_weights = neural_network.get_weights()
        elif validation_error < min_validation_error:
            min_validation_error = validation_error
            optimal_weights = neural_network.get_weights()

    # the weights of the network are updated to their optimal value, which has been determined during training
    for i in range(len(neural_network.layers)):
        neural_network.layers[i].W = optimal_weights[i]

    return training_error_history, validation_error_history

