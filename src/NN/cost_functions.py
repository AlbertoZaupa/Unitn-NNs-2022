import numpy as np
import abc


# The abstract class that defines the API of cost functions
class CostFunction(abc.ABC):

    @abc.abstractmethod
    def forward_pass(self, a, y):
        pass

    @abc.abstractmethod
    def backward_pass(self):
        pass


# The Mean Squared Error cost function
class MSE(CostFunction):

    def forward_pass(self, A, Y):
        """
        a: the predictions of the neural network, a 1 x N vector, where N is the number of data points
        y: the labels of our data points, again a 1 x N vector
        """
        self.A = A  # we store A because it will be used to compute the backward pass
        self.Y = Y  # we store Y for the same reason
        return np.mean((A - Y)**2, axis=1)

    def backward_pass(self):
        """
        We should return the partial derivative of the loss function with respect to the prediction A
        """

        return 2*(self.A - self.Y)
