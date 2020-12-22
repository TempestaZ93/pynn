import numpy as np
import math

from numpy.core.numeric import Inf, NaN

def cross_entropy_loss(y, a):
    result=0;
    for i in range(0,len(y)):
        result=result+y[i]*math.log2(a[i])
    return -result


def d_cross_entropy_loss(y, a):
    return 0


def log_loss(y, a):
    return -(y*np.log(a) + (1-y)*np.log(1-a))


def d_log_loss(y, a):
    a[a>0.99] = 0.99
    return (a - y)/(a*(1 - a))


loss_functions = {
    'cross_entropy': [cross_entropy_loss, d_cross_entropy_loss],
    'log': [log_loss, d_log_loss]
}