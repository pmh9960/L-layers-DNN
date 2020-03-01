import numpy as np
import math
from time import time
from dnn import L_model_forward
from regularization import (
    compute_cost_with_regularization,
    L_model_backward_with_regularization,
)


# GRADED FUNCTION: initialize_adam


def initialize_adam(L, parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """

    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        ### START CODE HERE ### (approx. 4 lines)
        v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
        s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
    ### END CODE HERE ###

    return v, s


# GRADED FUNCTION: update_parameters_with_adam


def update_parameters_with_adam(
    parameters, grads, v, s, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8
):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary
    t = t + 1

    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        ### START CODE HERE ### (approx. 2 lines)
        v["dW" + str(l + 1)] = (
            beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        )
        v["db" + str(l + 1)] = (
            beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]
        )
        ### END CODE HERE ###

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1 ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1 ** t)
        ### END CODE HERE ###

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        ### START CODE HERE ### (approx. 2 lines)
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * (
            grads["dW" + str(l + 1)] ** 2
        )
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * (
            grads["db" + str(l + 1)] ** 2
        )
        ### END CODE HERE ###

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        ### START CODE HERE ### (approx. 2 lines)
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2 ** t)
        ### END CODE HERE ###

        # print(
        #     v_corrected["dW" + str(l + 1)]
        #     / (np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)
        # )
        # print(
        #     v_corrected["db" + str(l + 1)]
        #     / (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)
        # )
        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        ### START CODE HERE ### (approx. 2 lines)
        parameters["W" + str(l + 1)] = parameters[
            "W" + str(l + 1)
        ] - learning_rate * v_corrected["dW" + str(l + 1)] / (
            np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon
        )
        parameters["b" + str(l + 1)] = parameters[
            "b" + str(l + 1)
        ] - learning_rate * v_corrected["db" + str(l + 1)] / (
            np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon
        )
        ### END CODE HERE ###

    return parameters, v, s


# GRADED FUNCTION: random_mini_batches


def random_mini_batches(X, Y, mini_batch_size, seed):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size
    )  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, mini_batch_size * (k) : mini_batch_size * (k + 1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * (k) : mini_batch_size * (k + 1)]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, mini_batch_size * int(num_complete_minibatches) :]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * int(num_complete_minibatches) :]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def optimize_with_adam(hyperparameters, X, Y, parameters, print_cost):

    (
        layer_dims,
        learning_rate,
        num_epochs,
        mini_batch_size,
        lambd,
        beta1,
        beta2,
        epsilon,
    ) = hyperparameters

    L = len(layer_dims) - 1
    m = X.shape[1]
    costs = []
    v, s = initialize_adam(L, parameters)

    for i in range(num_epochs):

        seed = np.random.seed(int(time()))
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:

            (minibatch_X, minibatch_Y) = minibatch
            # Forward propagation
            AL, caches = L_model_forward(L, minibatch_X, parameters)
            # Compute cost
            cost_total += compute_cost_with_regularization(
                layer_dims, AL, minibatch_Y, parameters, lambd
            )

            # Backward propagation
            grads = L_model_backward_with_regularization(AL, minibatch_Y, caches, lambd)

            # Update parameters
            parameters, v, s = update_parameters_with_adam(
                parameters, grads, v, s, i, learning_rate,
            )

        cost_avg = cost_total / (
            int(m / mini_batch_size) + int(m % mini_batch_size != 0)
        )
        if i % 100 == 0 and print_cost:
            print(f"Cost after Epoch {i} : {cost_avg}")
            costs.append(cost_avg)

    return parameters, grads, costs
