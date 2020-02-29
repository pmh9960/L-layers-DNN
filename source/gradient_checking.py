# Packages
import numpy as np
from lr_utils import *
from gc_utils import dictionary_to_vector, vector_to_dictionary, gradients_to_vector
from dnn import L_model_forward
from improved_dnn import L_model_backward_with_regularization


# GRADED FUNCTION: gradient_check_n


def gradient_check_n(layer_dims, parameters, gradients, X, Y, epsilon=1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """

    # Set-up variables
    parameters_values = dictionary_to_vector(layer_dims, parameters)
    grad = gradients_to_vector(layer_dims, gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    L = len(layer_dims) - 1

    # Compute gradapprox
    for i in range(num_parameters):

        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        ### START CODE HERE ### (approx. 3 lines)
        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2
        J_plus[i], _ = L_model_forward(
            L, X, vector_to_dictionary(layer_dims, thetaplus)
        )  # Step 3
        ### END CODE HERE ###

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus = np.copy(parameters_values)  # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2
        J_minus[i], _ = L_model_forward(
            L, X, vector_to_dictionary(layer_dims, thetaminus)
        )  # Step 3
        ### END CODE HERE ###

        # Compute gradapprox[i]
        ### START CODE HERE ### (approx. 1 line)
        gradapprox[i] = (J_plus[i] - J_minus[i]) / 2 / epsilon
        ### END CODE HERE ###

    # Compare gradapprox to backward propagation gradients by computing difference.
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(gradapprox - grad)  # Step 1'
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)  # Step 2'
    difference = numerator / denominator  # Step 3'
    ### END CODE HERE ###

    if difference > 2e-7:
        print(
            "\033[93m"
            + "There is a mistake in the backward propagation! difference = "
            + str(difference)
            + "\033[0m"
        )
    else:
        print(
            "\033[92m"
            + "Your backward propagation works perfectly fine! difference = "
            + str(difference)
            + "\033[0m"
        )

    return difference


def gradient_check(layer_dims, lambd, train_set_x, train_set_y, parameters):
    vector_x = train_set_x[:, 0, np.newaxis]
    vector_y = train_set_y[:, 0, np.newaxis]

    AL, caches = L_model_forward(len(layer_dims) - 1, vector_x, parameters)
    grads = L_model_backward_with_regularization(AL, vector_y, caches, lambd)
    difference = gradient_check_n(layer_dims, parameters, grads, vector_x, vector_y)
