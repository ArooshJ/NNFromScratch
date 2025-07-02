# Neural Networks and Backpropagation from Scratch in Python

This repository contains a Python implementation of a simple neural network and the backpropagation algorithm built entirely from scratch, without relying on high-level deep learning frameworks like TensorFlow or PyTorch. The primary goal of this project is to provide a clear and intuitive understanding of the fundamental concepts behind neural networks and how the backpropagation algorithm enables them to learn.

## Project Overview

The project is structured as a Jupyter notebook that guides the user through the process of building a neural network step-by-step. It starts with the most basic building block, a `Value` class for automatic differentiation, and progressively constructs more complex components like neurons, layers, and the complete neural network architecture.

### Key Features and Components Implemented:

* **Automatic Differentiation (`Value` Class)**:
    This is the core of the backpropagation implementation. The `Value` class wraps a scalar value and keeps track of its dependencies (children). It overloads standard arithmetic operators (`+`, `*`, `-`, `**`) and implements the `tanh` activation function. Crucially, each `Value` object stores its gradient (`grad`) and a `_backward` function that computes the gradient of the current `Value` with respect to its children.

* **Topological Sort**:
    A topological sort is used to determine the correct order of operations for backpropagation, ensuring that the gradients of parent nodes are computed before the gradients of their children.

* **Backward Pass (`backwardpass` method)**:
    This method initiates the backpropagation process from the output node (usually the loss). It sets the gradient of the output to 1 and then traverses the computational graph in reverse topological order, calling the `_backward` function for each node to accumulate gradients.

* **Zero Gradient (`zero_grad` method)**:
    Resets the gradients of all `Value` objects in the computational graph to zero, which is essential before computing gradients for a new training example or batch.

* **Neuron Class**:
    Represents a single artificial neuron. It initializes weights and a bias (either randomly, with ones, or with zeros) and implements a `forward` method to compute the weighted sum of inputs plus the bias.

* **Layer Class**:
    Represents a layer of neurons. It takes the number of inputs and the number of neurons in the layer as parameters and creates a list of `Neuron` objects. Its `forward` method processes a list of input `Value` objects through each neuron in the layer.

* **ANN Class**:
    Represents the complete Artificial Neural Network. It is composed of multiple `Layer` objects. The `forward` method passes the input through each layer, applying the `tanh` activation function after each hidden layer. The `update` method performs a simple stochastic gradient descent step by updating the weights and biases of all neurons based on their computed gradients and a learning rate.

* **Squared Error Loss Function (SELoss)**:
    A simple loss function implemented using the `Value` class, calculating the squared difference between the predicted output and the true label.

## Experiments and Analysis

The notebook includes several experiments to demonstrate the functionality of the implemented neural network and the impact of hyperparameters like the learning rate and the number of hidden layers. The `make_moons` dataset from `sklearn` is used for binary classification.

* **Initial Predictions**:
    The model's accuracy is evaluated before any training, showcasing the performance of the randomly initialized weights.

* **Model 1 (1 Hidden Layer, LR = 0.01)**:
    Training with a single hidden layer and a learning rate of 0.01 shows that the model can learn, with accuracy increasing and loss decreasing over epochs, albeit with some oscillations.

* **Model 2 (1 Hidden Layer, LR = 0.1)**:
    Increasing the learning rate to 0.1 with a single hidden layer leads to faster changes in weights and potentially quicker convergence, but also more significant oscillations in accuracy and loss.

* **Model 3 (2 Hidden Layers, LR = 0.01)**:
    Adding a second hidden layer while keeping the learning rate at 0.01 results in smoother loss curves and improved accuracy compared to the single-hidden-layer model with the same learning rate. This suggests that a deeper network can better capture the non-linear nature of the data.

* **Model 4 (2 Hidden Layers, LR = 0.000001)**:
    Using a very low learning rate with two hidden layers demonstrates extremely slow learning, with minimal changes in accuracy and high loss values. This highlights the importance of an appropriate learning rate for effective training.

* **Model 5 (2 Hidden Layers, LR = 0.1)**:
    Training with a higher learning rate (0.1) and two hidden layers shows signs of overfitting. The model performs well on the training data (implied by the decreasing loss initially and potentially increasing training accuracy if plotted), but its performance on the test set (accuracy) plateaus or even decreases after a certain number of epochs, while the test loss may start to increase again. This indicates that the model is learning the training data too well, including the noise, and is not generalizing well to unseen data.

## Limitations and Future Work

This implementation serves as a valuable educational tool for understanding the core concepts of neural networks and backpropagation. However, it has several limitations compared to production-ready deep learning libraries:

* **No Vectorization**:
    The current implementation processes inputs and performs calculations element-wise. This is computationally inefficient compared to vectorized operations on tensors, which are heavily utilized in libraries like NumPy, PyTorch, and TensorFlow for faster matrix multiplications and other operations.

* **Stochastic Gradient Descent Only**:
    The training loop currently implements stochastic gradient descent (SGD), where weights are updated after processing each individual training example. Implementing batch gradient descent would require accumulating gradients over a batch of examples before performing a single weight update. While technically possible with the current `Value` class by summing the losses of a batch, it would lead to a large and complex computational graph, further highlighting the need for vectorization.

* **Limited Functionality**:
    The implementation only includes basic arithmetic operations and the `tanh` activation. A full-fledged deep learning library supports a wide range of layers, activation functions, loss functions, and optimization algorithms.

Future work could involve:

* Implementing vectorized operations using libraries like NumPy to improve computational efficiency.
* Adding support for batch gradient descent.
* Implementing more activation functions (e.g., ReLU, Sigmoid).
* Adding support for different loss functions (e.g., Cross-Entropy Loss).
* Implementing more advanced optimization algorithms (e.g., Adam, RMSprop).

## Attribution

This project was heavily inspired by and refers to concepts learned from Andrej Karpathy's excellent micrograd video series and associated code, which provides a deep dive into building automatic differentiation and neural networks from scratch.
