import numpy as np
from activation.functions import activation_functions
import random

class Layer(object):

    def __init__(self, inputs: int, neurons: int, activation: str):
        self.W = np.random.randn(neurons, inputs)
        
        self.W = np.zeros((neurons, inputs), dtype=np.float64)
        for row in range(self.W.shape[0]):
            for col in range(self.W.shape[1]):
                self.W[row, col] = random.uniform(0, 1)
        
        self.b = np.random.randn(neurons, 1)
        self.activation = activation
        act = activation_functions.get(activation)
       
        self.act = act[0]
        self.d_act = act[1]

    def __repr__(self):
        return "Layer:(neurons: {}, inputs: {}, activation function: {})".format(np.size(self.W, 0), np.size(self.W, 1), self.activation)

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return np.size(self.W, 0)

    def feedforward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev) + self.b

        zMax = np.max(self.Z) / 700
        self.Z = self.Z / zMax

        self.A = self.act(self.Z)
        return self.A

    # This method takes in an error and a learning rate
    # After calculating and applying the weight adjustmends it returns the error for the previous layer
    def optimize(self, dA, learning_rate):
        dZ = np.multiply(self.d_act(self.Z), dA)
        dW = 1/dZ.shape[1] * np.dot(dZ, self.A_prev.T)
        db = 1/dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db

        return dA_prev