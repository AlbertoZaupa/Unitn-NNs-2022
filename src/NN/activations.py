import numpy as np
import abc


# The abstract class that defines the API of activation functions
class ActivationFunction(abc.ABC):

    @abc.abstractmethod
    def forward_pass(self, Z):
        pass

    @abc.abstractmethod
    def backward_pass(self, dLdA):
        pass


# The Rectified Linear Unit activation function
class ReLU(ActivationFunction):

    def forward_pass(self, Z):
        """
        z: the pre-activation values (which are the results of the scalar products)

        ReLU performs max(0, input_value)
        """
        Z[Z < 0] = 0
        self.A = Z
        return Z

    def backward_pass(self, dLdA):
        """
        dLdA: the partial derivative of the loss function with respect to the output of the activation function
        We should return the partial derivative of the loss function with respect to the inputs of the activation function,
        which we call Z. Therefore we should return dLdZ.

        The chain rule states that: dLdZ = dAdZ * dLdA.

        dAdZ is a diagonal matrix for every vector in the matrix A: if we consider one by
        one the vectors that make up the matrices A and Z, for every element of a vector in A we should compute
        its derivatives with respect to all the elements of the corresponding vector in Z,
        so for every element of a vector of A we get a vector of derivatives. Luckily only when considering the
        element of the vector of Z that has the same index of the element of the vector of A
        we get a derivative that is different from 0.

        It follows that as a whole dAdZ is a 3-dimensional tensor, of dimensions (n x n x N). We are lucky that the
        N matrices of dAdZ are diagonal, therefore the result of the tensor product dAdZ * dLdA is equal to
        dLdA @ f'(Z), where f is the activation function that produces A and the symbol @ indicates element-wise
        multiplication.

        To convince yourself that the proposition just stated is true, try to visualize the tensor
        product as a sequence of N matrix-to-vector multiplications taking into consideration that dAdZ is
        composed of diagonal matrices.
        """

        return dLdA * (self.A > 0)


# the identity activation function. It is used by the last layer of a network
class Identity(ActivationFunction):

    def forward_pass(self, Z):
        return Z

    def backward_pass(self, dLdA):
        return dLdA
