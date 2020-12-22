from loss.functions import *
from model.layer import Layer
import random

class Model(object):

    def __init__(self, layers, loss, learning_rate, regL1, regL2):
        self.layers = []
        for idx in range(0, len(layers)):
            self.layers.append(Layer(layers[idx][0], layers[idx][1], layers[idx][2]))

        self.loss_name = loss
        loss_arr = loss_functions.get(loss)
        self.loss = loss_arr[0]
        self.d_loss = loss_arr[1]

        self.learning_rate = learning_rate
        self.regL1 = regL1
        self.regL2 = regL2


    def __repr__(self):
        return "Layer: {}\nLoss function: {}\nOptimizer:{}".format(self.layers, self.loss_name, self.optimizer_name)

    
    def feedforward(self, X):
        A = X
        for layer in self.layers:
            A = layer.feedforward(A)

        return A


    def _next_batch(self, X, y, batch_size):
        for i in np.arange(0, X.shape[0], batch_size):
            # yield a tuple of the current batched data and labels
            
            start = int(random.uniform(0, X.shape[0] - batch_size))
            if i + batch_size >= X.shape[0]-1:
                yield (X[start:X.shape[0]], y[start:X.shape[0]])
            else:
                yield (X[start:start + batch_size], y[start:start + batch_size])


    def _optimize(self, dA):
        for layer in reversed(self.layers):
            dA = layer.optimize(dA, self.learning_rate)


    def train(self, X, y, epochs, batch_size):
        for epoch in range(epochs):
            batch = 0
            for (batch_X, batch_y) in self._next_batch(X, y, batch_size):

                curr_batch_size = batch_X.shape[0]
                dA = np.zeros((len(self.layers[-1]), 1))


                weight_sum : np.float64 = 0
                weight_sum_sq : np.float64 = 0
                for layer in self.layers:
                    weight_sum += np.sum(np.abs(layer.W))
                    weight_sum_sq += np.sum(layer.W**2)

                loss = 0
                for i in range(curr_batch_size):
                    c_X = batch_X[i]
                    
                    c_y = np.zeros((len(self.layers[-1]), 1))
                    c_y[batch_y[i]][0] = 1
                    
                    A = self.feedforward(c_X)
                    loss = self.d_loss(c_y, A)
                    dA += loss

                    print(epoch + 1, ":", batch + 1, "-", i + 1, "          ", end='\r')
                
                
                dA = dA / curr_batch_size
                dA +=  + self.regL1 * weight_sum + self.regL2 * weight_sum_sq

                self._optimize(dA)
                batch = batch + 1

        print("")

