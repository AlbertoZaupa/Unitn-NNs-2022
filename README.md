# Unitn-NNs-2022

In this repository I have implemented a simple feed-forward Neural Network, the Backpropagation algorithm used to 
compute the gradients of the loss function with respect to the weights of the network, and mini-batch SGD, which is 
then used to actually train the network. 

A simple example dataset is used to demonstrate that the neural network is actually capable of learning unknown mappings
from input feature vectors, to desired outputs.

Along with the implementation of the NN, I have uploaded some notes in which I present the topic of training neural networks.
In these notes I dive into the details of the techniques used to compute the gradients necessary for Gradient Descent.
Still, to fully implement Backpropagation some additional details, that I do not cover in the notes, have to be taken into
consideration. 

If you are interested in implementing Backprop yourself, I suggest you to get to the point where you fully understand the
concepts presented in the notes (you'll find plenty of additional material online). Then if you'll need some inspiration, have a look at the code
in this repo. I have included some comments that should clarify those details left out from the notes.