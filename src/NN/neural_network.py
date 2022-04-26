from src.NN.layer import Layer
from src.NN.cost_functions import CostFunction


# The class that wraps layers of neurons into a fully-connected sequential Neural Network
class NN:

    input_dimension: int  # the length of the data vectors
    layers: list  # the list of layers
    cost_function: CostFunction  # the cost function used to train the model

    def __init__(self, input_dimension, cost_function):
        self.input_dimension = input_dimension
        self.cost_function = cost_function
        self.layers = []

    def add_layer(self, output_dimension, activation_function):
        """
        This method just adds a layer to our neural network
        """

        # first the input dimension of the layer is computed
        input_dimension = 0
        if len(self.layers) == 0:
            input_dimension = self.input_dimension
        else:
            input_dimension = self.layers[-1].output_dimension

        # then the layer is added to the list of layers of the network
        self.layers.append(Layer(input_dimension, output_dimension, activation_function))

    def predict(self, X, Y=None):
        """
        This method takes in a data vector, or a batch of data vectors, and outputs the prediction of the model.
        """

        for layer in self.layers:
            # the output of each layer is passed as input to the following one
            X = layer.forward_pass(X)

        if Y is None:
            # the last output is the prediction of the neural network
            return X

        # if an expectation is given, the error is computed
        return X, self.cost_function.forward_pass(X, Y)

    def train(self, X, Y, nabla):
        """
        X: batch of data on which this training iteration will be executed
        nabla: the gradient update step coefficient for this training iteration

        This method performs a forward pass and a backward pass on the batch of data. Then the weights of all the
        layers of the network are updated.

        The batch of data is in general matrix, but the method works correctly also when a single data vector is
        used for a training iteration, like for example if we are using stochastic gradient descent. In that case the
        matrix X will have dimensions (m x 1), which means it actually is a vector.
        """

        n_layers = len(self.layers)

        # the forward pass
        A, error = self.predict(X, Y)

        # the derivative of the loss function with respect to the NN's prediction is computed
        dLdA = self.cost_function.backward_pass()

        # the backward pass
        for i in range(n_layers):
            # the derivative of the loss function with respect to the input of each layer is passed to the previous one
            # as the derivative of the loss function with respect to its output
            dLdA = self.layers[n_layers-1-i].backward_pass(dLdA)

        # the weights are updated
        for layer in self.layers:
            layer.gradient_update(nabla)

        return error

    def get_weights(self):
        """
        this method is used to get the current weights of the model
        """

        return [layer.W for layer in self.layers]
