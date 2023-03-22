import numpy as np
import requests, gzip, os, hashlib

"""
Script for download MNIST and saving in expected place as .npy files.

Adapted from code in 
https://towardsdatascience.com/mnist-handwritten-digits-classification-from-scratch-using-python-numpy-b08e401c4dab
"""

def fetch(url):
    data = requests.get(url).content
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]


np.save("mnist_train_data.npy", X)
np.save("mnist_test_data.npy", X_test)
np.save("mnist_train_labels.npy", Y)
np.save("mnist_test_labels.npy", Y_test)
