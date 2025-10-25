import numpy as np
import matplotlib.pyplot as plt # optional for visualizing

import urllib.request
import os

# Download MNIST
mnist_path = "mnist.npz"

if not os.path.exists(mnist_path): # checks if data is already here
    print("Downloading MNIST...")
    urllib.request.urlretrieve(
        "https://s3.amazonaws.com/img-datasets/mnist.npz",
        mnist_path
    ) # retrieves data and places it at path
    print("Download complete.")

# ~~~~~ Load MNIST ~~~~~

# to display the keys inside of the NpzFile object:
#       data1 = np.load("mnist.npz")
#       print(data1.files)

# 'with...as' used to automatically manage resources & handle cleanup
# implements error handling behind the scenes
# np.load creates an NpzFile object that acts like a dictionary
# keys correspond to numpy arrays, which we extract
with np.load(mnist_path) as data: # context manager
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

# flatten and normalize images
# initially, each image is 28x28 px stored as 2D array of ints 0-255
# 0 = black, 255 = white
# 60k samples, so x_train is 3D array of size (60000, 28, 28)
# -> x_train.shape[0] = 60k
# we specify this first dimension in reshape, -1 autofills the second
# each image grid becomes a 784x1 vector
# NNs work best when inputs are small continuous values, so we normalize (divide by 255)
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# optional, visualize an image
# imshow takes numpy array and visualizes as px intensity vals
# must reshape the first image back into 2d for imshow
"""
plt.imshow(x_train[0].reshape(28,28), cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()
"""

# ~~~~~ Prepare the labels ~~~~~

# one hot encoding
# np.eye creates an identity matrix of nxn
# we then index into it with y_train using fancy indexing
# this slices out and orders the corresponding lines of the identity matrix that match each element of y_train
# y_train.shape = (60000, 10)
# where each row represents a label with a one hot vector
num_classes = 10
y_train_encoded = np.eye(num_classes)[y_train]
y_test_encoded = np.eye(num_classes)[y_test]

# ~~~~~ Structure the NN ~~~~~

# we do a fully connected feedforward network, 1 hidden layer
# input: 784 neurons
# hidden: 128 neurons
# output: 10 neurons
# ReLU for hidden layer activation and softmax for output

# init weights and biases
# * 0.01 just makes weights small, starts training nice
# biases init to zero
# weight matrix.shape = (input neurons, neurons in layer)
# bias matrix.shape = (1, # of neurons in layer)
# biases are broadcasted to each row during addition

# np.random.randn(784, 128) generates matrix of shape (a,b)
# with values sampled from standard normal dist, mean=0 stdev=1
# we do this to ensure reasonable starting variation without bias
# np.zeros creates a row vector of zeros

# hidden layer
w1 = np.random.randn(784, 128) * 0.01 # weights
b1 = np.zeros((1, 128))               # biases
# output layer
w2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))

# forward pass
# z is a 1x128 vector with elements representing pre-activation of neurons
# z = xw + b
# x_train.shape = (60k, 784); (xtrain dot w1).shape = (60k, 128)
# ReLU = rectified linear unit, zeroes out negative values
# z1.shape = a1.shape
z1 = np.dot(x_train, w1) + b1    # linear step for hidden layer
a1 = np.maximum(0, z1)

# output layer
# z = aw + b
# softmax converts logits into probabilities that we can compare to one hot labels
z2 = np.dot(a1, w2) + b2    # shape = (60k, 10), contains logits
exp_scores = np.exp(z2)    # exponentiate
a2 = exp_scores / np.sum(exp_scores, axis =1, keepdims=True)
# ^ divide by sum of row to normalize for probability
# ^ axis=1 is sum across columns, keepdims prevents dimension collapse

# ~~~~~ Cross entropy loss ~~~~~

# - how well do the predicted probabilities match true labels?
# - cross entropy - how different two probability distributions are
# - only log prob of the correct class contributes
# - penalty for assigning low probability to the right class
# - loss = scalar loss measuring performance
# - log amplifies differences near zero + allows addition of probs
# - must negate because log of 0<x<1 is negative
loss = -np.mean(np.sum(y_train_encoded * np.log(a2), axis=1))

# ~~~~~ Backprop ~~~~~

# - dz2 = derivative of loss w.r.t. z2 before softmax aka output error
# - cross entropy and softmax simply derivative to a - y
dz2 = a2 - y_train_encoded    # shape = (batch_size, 10)






#if __name__ == "__main__":
