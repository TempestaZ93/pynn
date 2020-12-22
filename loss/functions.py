import numpy as np

def cross_entropy_loss(y, a):
    return np.multiply(y, -np.log(a)).sum(1)

def d_cross_entropy_loss(y, a):
    a[y.argmax()] -= 1
    return a


def log_loss(y, a):
    return -(y*np.log(a) + (1-y)*np.log(1-a))


def d_log_loss(y, a):
    a[a>0.99] = 0.99
    return (a - y)/(a*(1 - a))


loss_functions = {
    'cross_entropy': [cross_entropy_loss, d_cross_entropy_loss],
    'log': [log_loss, d_log_loss]
}