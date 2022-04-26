from src.datasets import dataset1
from src.NN.neural_network import NN
from src.NN.cost_functions import MSE
from src.NN.activations import ReLU, Identity
from src.NN.learning_algorithms import mini_batch_gradient_descent
from src.NN.utils import display_history

(X_train, Y_train, X_val, Y_val) = dataset1.dataset()

neural_network = NN(2, MSE())
neural_network.add_layer(10, ReLU())
neural_network.add_layer(1, Identity())

training_error_history, validation_error_history = \
    mini_batch_gradient_descent(X_train, Y_train, X_val, Y_val, neural_network, int(dataset1.DATASET_SIZE/100), 100)

display_history(training_error_history, validation_error_history)
