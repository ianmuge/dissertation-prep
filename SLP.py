import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from random import random


class SLP:
    def __init__(self,n_inputs, n_hidden, n_outputs):
        network = list()
        hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
        output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        self.error,self.epoch,self.network=[],[],network

    @jit(nopython=True, parallel=True)
    def activate(self,weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    def transfer(self,activation):
        return 1.0 / (1.0 + np.exp(-activation))

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    @jit(nopython=True, parallel=True)
    def forward_propagate(self,network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    @jit(nopython=True, parallel=True)
    def backward_propagate_error(self,network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    @jit(nopython=True, parallel=True)
    def update_weights(self,network, row, l_rate):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']

    @jit(nopython=True, parallel=True)
    def train_network(self,network, train, l_rate, n_epoch, n_outputs):
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(network, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(network, expected)
                self.update_weights(network, row, l_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
            self.error.append(sum_error)
            self.epoch.append(epoch)
        plt.plot(self.epoch,self.error)
        plt.ylabel('Error')
        plt.xlabel('Epochs')
        plt.title("Error Function")
        plt.show()

    @jit(nopython=True, parallel=True)
    def predict(self,network, row):
        outputs = self.forward_propagate(network, row)
        return outputs.index(max(outputs))
