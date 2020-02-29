import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
from dnn import *


# GRADED FUNCTION: initialize_parameters_he


def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    parameters = {}
    L = len(layers_dims) - 1  # integer representing the number of layers

    for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters["W" + str(l)] = np.random.randn(
            layers_dims[l], layers_dims[l - 1]
        ) * np.sqrt(2 / layers_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###

    return parameters


# GRADED FUNCTION: compute_cost_with_regularization


def compute_cost_with_regularization(layer_dims, AL, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization. See formula (2) above.
    
    Arguments:
    AL -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]
    L = len(layer_dims) - 1
    sum_weight = 0
    W = {}

    for l in range(L):
        W[l + 1] = parameters["W" + str(l + 1)]

    cross_entropy_cost = compute_cost(
        AL, Y
    )  # This gives you the cross-entropy part of the cost

    ### START CODE HERE ### (approx. 1 line)
    for l in range(L):
        sum_weight += np.sum(np.square(W[l + 1]))
    L2_regularization_cost = lambd / 2.0 / m * sum_weight
    ### END CODER HERE ###

    cost = cross_entropy_cost + L2_regularization_cost

    return cost


def L_model_backward_with_regularization(AL, Y, caches, lambd):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ... 
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL
    W = {}

    for l in range(L):
        current_cache = caches[l]
        linear_cache, activation_cache = current_cache
        W[l + 1] = linear_cache[1]

    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    ### END CODE HERE ###

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[L - 1]
    (
        grads["dA" + str(L - 1)],
        grads["dW" + str(L)],
        grads["db" + str(L)],
    ) = linear_activation_backward(dAL, current_cache, "sigmoid")

    grads["dW" + str(L)] += lambd / m * W[L]
    ### END CODE HERE ###

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu"
        )
        grads["dA" + str(l)] = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu"
        )[0]
        grads["dW" + str(l + 1)] = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu"
        )[1]
        grads["db" + str(l + 1)] = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu"
        )[2]
        grads["dW" + str(l + 1)] += lambd / m * W[l + 1]
        ### END CODE HERE ###

    return grads


def optimize(hyperparameters, train_set_x, train_set_y, parameters, print_cost):

    layer_dims, lambd, learning_rate, iterations, seed = hyperparameters
    L = len(layer_dims) - 1
    costs = []

    for i in range(iterations):
        # Forward propagation
        AL, caches = L_model_forward(L, train_set_x, parameters)
        # Compute cost
        cost = compute_cost_with_regularization(
            layer_dims, AL, train_set_y, parameters, lambd
        )

        # Backward propagation
        grads = L_model_backward_with_regularization(AL, train_set_y, caches, lambd)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0 and print_cost:
            print(f"Cost after iteration {i} : {cost}")
            costs.append(cost)

    return parameters, grads, costs


def predict(L, X, parameters):
    AL, caches = L_model_forward(L, X, parameters)
    print(AL)
    Y_predicted = np.zeros((1, AL.shape[1]))
    for i in range(AL.shape[1]):
        if AL[0][i] > 0.5:
            Y_predicted[0][i] = 1
        else:
            Y_predicted[0][i] = 0

    assert Y_predicted.shape == (1, AL.shape[1])

    return Y_predicted
