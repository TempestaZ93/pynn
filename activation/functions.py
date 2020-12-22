import numpy as np

def softmax(x):
    ex = np.exp(x)
    return ex/ex.sum()


def d_softmax(x):
    ex=np.exp(x)
    ex_sum=ex.sum()
    return ex/ex_sum*(1-ex/ex_sum)


def relu(x):
    p=x;
    for i in range(0,len(x)):
        p[i]=max(0,x[i])
    return p


def d_relu(x):
    p=x;
    for i in range(0,len(x)):
        if x[i] == 0:
            x[i] = 0.001
        p[i]=max(0,x[i]/abs(x[i]))
    return p


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.square(np.tanh(x))


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def d_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)


activation_functions = {
    'softmax': [softmax, d_softmax],
    'relu': [relu, d_relu],
    'tanh': [tanh, d_tanh],
    'sigmoid': [sigmoid, d_sigmoid]
}