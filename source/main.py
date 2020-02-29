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
from adam import optimize_with_adam

current_time = time.time()
current_time_str = time.strftime("%Y%m%d_%H%M", time.localtime(current_time))
current_time_str = current_time_str[2:]
seed = int(current_time)
np.random.seed(seed)

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

# Hyperparameters
layer_dims = [train_set_x.shape[0], 3, train_set_y.shape[0]]
learning_rate = 0.01
iterations = 2000
lambd = 0.1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
hyperparameters = (layer_dims, learning_rate, iterations, lambd, beta1, beta2, epsilon)

# Initialize
parameters = initialize_parameters_he(layer_dims)

# opened_file_hyperparameters = dir / "20200229_1150_hyperparameters.txt"
# opened_file_parameters = dir / "20200229_1150_parameters.txt"
# opened_file_costs = dir / "20200229_1150_costs.txt"
# with open(opened_file_hyperparameters, "rb") as f:
#     hyperparameters = pickle.loads(f)
# with open(opened_file_parameters, "rb") as f:
#     parameters = pickle.loads(f)
# with open(opened_file_costs, "rb") as f:
#     prev_costs = pickle.loads(f)
# layer_dims, lambd, learning_rate, iterations, seed = hyperparameters

print(parameters)

# Optimize
parameters, grads, costs = optimize_with_adam(
    hyperparameters, train_set_x, train_set_y, parameters, print_cost=True,
)

# costs = prev_costs + costs

# # Grad Chek
# gradient_check(layer_dims, lambd, train_set_x, train_set_y, parameters)

# Plot
plt.xlabel("iteration (hundred)")
plt.ylabel("cost")
plt.plot(costs)
# Predict
Y_predicted = predict(len(layer_dims) - 1, test_set_x, parameters)
accuracy = 100 - np.mean(np.abs(Y_predicted - test_set_y)) * 100

result = {"Y_predicted": Y_predicted, "accuracy": accuracy}

# Save results
root = os.getcwd()
dir = root + "/results/" + current_time_str + "/"
access_rights = 0o755
os.mkdir(dir, access_rights)

plt.savefig(dir + current_time_str + ".png", dpi=300)

with open(dir + current_time_str + "_parameters.pkl", "wb") as f:
    pickle.dump(parameters, f)
with open(dir + current_time_str + "_costs.pkl", "wb") as f:
    pickle.dump(costs, f)
with open(dir + current_time_str + "_hyperparameters.pkl", "wb") as f:
    pickle.dump(hyperparameters, f)
with open(dir + current_time_str + "_hyperparameters.txt", "wt") as f:
    f.write("Hyperparameters\n\n")
    f.write("Seed : " + str(seed) + "\n")
    f.write("Iteration : " + str(iterations) + "\n")
    f.write("Layers dimensions : " + str(layer_dims) + "\n")
    f.write("Learning rate : " + str(learning_rate) + "\n")
    f.write("Regularization factor Lambda : " + str(lambd) + "\n")
    f.write("Beta1 : " + str(beta1) + "\n")
    f.write("Beta2 : " + str(beta2) + "\n")
    f.write("Epsilon : " + str(epsilon) + "\n")
# with open(dir / current_time_str + "_parameters" + with_open + ".pkl", "wb") as f:
#     pickle.dump(parameters, f)
# with open(dir / current_time_str + "_costs" + with_open + ".pkl", "wb") as f:
#     pickle.dump(costs, f)
# with open(dir / current_time_str + "_hyperparameters" + with_open + ".pkl", "wb") as f:
#     pickle.dump(hyperparameters, f)
with open(dir + current_time_str + "_result_" + str(accuracy) + ".pkl", "wb") as f:
    pickle.dump(result, f)

