import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0, x)

    return s


def dictionary_to_vector(layer_dims, parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    cnt = 0
    L = len(layer_dims) - 1

    for l in range(L):
        for key in ["W" + str(l + 1), "b" + str(l + 1)]:
            new_vector = np.reshape(parameters[key], (-1, 1))

            if cnt == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)

            cnt += 1

    return theta


def vector_to_dictionary(layers_dims, theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    L = len(layers_dims) - 1
    start = end = 0

    for l in range(L):
        row = int(layers_dims[l + 1])
        col = int(layers_dims[l])

        start = end
        end = start + row * col

        parameters["W" + str(l + 1)] = theta[start:end].reshape((row, col))

        start = end
        end = end + row

        parameters["b" + str(l + 1)] = theta[start:end].reshape((row, 1))

    return parameters


def gradients_to_vector(layer_dims, gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    cnt = 0
    L = len(layer_dims) - 1

    for l in range(L):
        for key in ["dW" + str(l + 1), "db" + str(l + 1)]:
            new_vector = np.reshape(gradients[key], (-1, 1))

            if cnt == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)

            cnt += 1

    return theta
