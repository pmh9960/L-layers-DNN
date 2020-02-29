import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import time
import pickle
from pathlib import Path
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from improved_dnn import *
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

root = Path(".")
dir = root / "source" / "results"

# Initialize
hyperparameters_dir = dir / "20200229_1300_hyperparameters.pkl"
parameters_dir = dir / "20200229_1300_parameters.pkl"
costs_dir = dir / "20200229_1300_costs.pkl"
with open(hyperparameters_dir, "rb") as f:
    hyperparameters = pickle.load(f)
with open(parameters_dir, "rb") as f:
    parameters = pickle.load(f)
with open(costs_dir, "rb") as f:
    costs = pickle.load(f)

layer_dims, lambd, learning_rate, iterations, seed = hyperparameters


# # Grad Chek
# gradient_check(layer_dims, lambd, train_set_x, train_set_y, parameters)

# # Plot
# plt.xlabel("iteration (hundred)")
# plt.ylabel("cost")
# plt.plot(costs)
# plt.show()


## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "1.jpg"  # change this to the name of your image file
## END CODE HERE ##

# We preprocess the image to fit your algorithm.
fname = root / "images" / my_image
image = np.array(ndimage.imread(fname, flatten=False))
image = image / 255.0
my_image = (
    scipy.misc.imresize(image, size=(num_px, num_px))
    .reshape((1, num_px * num_px * 3))
    .T
)
plt.imshow(image)
plt.show()
# Predict
Y_predicted = predict(len(layer_dims) - 1, my_image, parameters)
print(Y_predicted)


# # Predict
# Y_predicted = predict(len(layer_dims) - 1, train_set_x, parameters)
# print(Y_predicted)
# print(train_set_y)
# accuracy = 100 - np.mean(np.abs(Y_predicted - train_set_y)) * 100
# print(accuracy)

# result = {"Y_predicted": Y_predicted, "accuracy": accuracy}

# result_dir = dir / "20200229_1300_result_train.pkl"
# with open(result_dir, "wb") as f:
#     pickle.dump(result, f)

