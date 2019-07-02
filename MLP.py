import winsound
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from random import random


class MLP:
    def __init__(self,epochs,n_input,n_hidden,n_output,batch_size=100,learn_rate=0.5):
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.n_output=n_output
        self.batch_size=batch_size

    def transfer_sig(self,x):
        return 1 / (1 + np.exp(-x))

    def transfer_der_sig(self,x):
        return x * (1 - x)

    def transfer_relu(self,x):
        x[x < 0] = 0
        return x

    def transfer_der_relu(self,x):
        x[x < 0] = 0
        x[x > 0] = 1
        return x

    @jit(nopython=True, parallel=True)
    def feed_forward(self):

    @jit(nopython=True, parallel=True)
    def back_propagate(self):

    def loss(self,y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    @jit(nopython=True, parallel=True)
    def train(self):
        for i in range(self.epochs):
            for j in range(x.shape[0] // self.batch_size):

mlp=MLP(epochs=10,learn_rate=0.3,
        n_input=784,n_hidden=(10,50,10),
        n_output=10,batch_size=100)