import numpy as np
from src.NN.activations import ActivationFunction


class Layer:

    activation_function: ActivationFunction  # the chosen activation function for the layer
    output_dimension: int  # the length of the output vector

    def __init__(self, m, n, activation_function):
        """
        m: the size of the input vectors
        n: the size of the output vector
        activation: the activation function for this layer

        A m x n matrix is initialized with random values
        """

        self.output_dimension = n
        self.W = np.random.normal(size=(m, n))  # the weight matrix of the layer
        self.b = np.zeros(shape=(n, 1))  # the bias vector of the layer
        self.activation_function = activation_function

    def forward_pass(self, X):
        """
        x: a (m x N) array, were N is the number of data points and m is equal to the number of rows of the
        weight matrix for this layer.
        """

        self.X = X  # the value of the inputs is saved for the backward pass
        Z = np.dot(np.transpose(self.W), X) + self.b  # the result is the pre-activation (n x N)
        return self.activation_function.forward_pass(Z)  # the activation is again (n x N)

    def backward_pass(self, dLdA):
        """
        dLdA: the partial derivative of the loss function with respect to the output of the layer. (n x N) matrix.
        We should return the derivative of the loss function with respect to the input of the layer. By using the
        chain rule: dLdX = dZdX * dLdZ = W * dLdZ.
        Dimensions check: (m x N) = (m x n) * (n x N)

        Because of the chain rule, dLdW = dZdW * transpose(dLdZ) = X * transpose(dLdZ)
        We have dLdA and by using the chain rule we can get: dLdZ = dAdZ * dLdA .
        dLdZ also has dimensions (n x N).
        Dimensions check: (m x N) * transpose( (n x N) ) = (m x N) * (N x n) = (m x n)

        To get dLdb, we would need to compute dZdb * dLdZ, but dZdb is equal to the identity matrix, so we can
        just sum over the columns of dLdZ
        """

        dLdZ = self.activation_function.backward_pass(dLdA)  # derivative of loss with respect to pre-activations
        self.dLdW = np.matmul(self.X, np.transpose(dLdZ))  # derivative of loss with respect to the weights
        self.dLdb = np.reshape(np.sum(dLdZ, axis=1), newshape=(dLdZ.shape[0], 1))  # derivative of loss wrt the biases
        return np.matmul(self.W, dLdZ)  # derivative of loss with respect to the output of the previous layer

    def gradient_update(self, nabla):
        """
        nabla: the step coefficient

        This method simply updates the weights of the layer after a forward and backward pass.
        """

        self.W -= nabla*self.dLdW
        self.b -= nabla*self.dLdb
