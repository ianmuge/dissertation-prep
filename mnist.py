import pandas as pd
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt

def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %d' % y[idx])
    plt.show()

train_images,train_labels = loadlocal_mnist(
    images_path='./data/train-images.idx3-ubyte',
    labels_path='./data/train-labels.idx1-ubyte')

test_images,test_labels  = loadlocal_mnist(
    images_path='./data/t10k-images.idx3-ubyte',
    labels_path='./data/t10k-labels.idx1-ubyte')
