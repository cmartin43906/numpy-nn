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

# Load MNIST

with np.load(mnist_path) as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

# flatten and normalize images
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# optional, visualize an image
plt.imshow(x_train[0].reshape(28,28), cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()


#if __name__ == "__main__":
