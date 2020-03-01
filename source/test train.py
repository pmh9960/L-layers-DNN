import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import pickle
import time
import os
from pathlib import Path
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from regularization import *
from gradient_checking import gradient_check_n, gradient_check

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
train_set_x = train_set_x_flatten / 255.0
test_set_x = test_set_x_flatten / 255.0

test_time = "200301_1703"

root = os.getcwd()
dir = root + "/results/" + test_time + "/"
# Initialize
hyperparameters_dir = dir + test_time + "_hyperparameters.pkl"
parameters_dir = dir + test_time + "_parameters.pkl"
costs_dir = dir + test_time + "_costs.pkl"
with open(hyperparameters_dir, "rb") as f:
    hyperparameters = pickle.load(f)
with open(parameters_dir, "rb") as f:
    parameters = pickle.load(f)
with open(costs_dir, "rb") as f:
    costs = pickle.load(f)

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
# Predict
Y_predicted = predict(len(layer_dims) - 1, train_set_x, parameters)
# print(Y_predicted)
# print(train_set_y)
accuracy = 100 - np.mean(np.abs(Y_predicted - train_set_y)) * 100
# print(accuracy)

result = {"Y_predicted": Y_predicted, "accuracy": accuracy}

with open(dir + test_time + "_train_result_" + str(accuracy) + ".pkl", "wb") as f:
    pickle.dump(result, f)

