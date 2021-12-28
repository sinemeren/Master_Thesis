import numpy as np
from numpy.ma import tanh as tanh_function, exp


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):

    return 1 / (1 + exp(-x))


def tanh(x):
    return tanh_function(x)
