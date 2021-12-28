import numpy as np
from sklearn import metrics


class RVFL:

    def __init__(self, numberOfNodes, seed, activation_function):

        self.numberOfNodes = numberOfNodes
        self.seed = seed
        self.activation_function = activation_function
        self.w = []
        self.random_weights = []

    def train(self, x, y):

        bias = np.ones([x.shape[0], 1])
        x = np.concatenate((x, bias), axis=1)

        np.random.seed(self.seed)
        self.random_weights = np.random.randn(
            np.shape(x)[1], self.numberOfNodes)

        H_2 = self.activation_function(
            x.dot(self.random_weights))
        H = np.concatenate((x, H_2), axis=1)

        self.w = np.linalg.pinv(
            H).dot(y)

    def predict(self, x):

        bias = np.ones([x.shape[0], 1])
        x = np.concatenate((x, bias), axis=1)
        H_2 = self.activation_function(
            x.dot(self.random_weights))
        H = np.concatenate((x, H_2), axis=1)
        predicted_output = H.dot(
            self.w)

        return predicted_output

    def eval(self, predictions, y):
        # mean squarred error error
        mse = metrics.mean_squared_error(y, predictions)

        # mean absolute error
        mae = metrics.mean_absolute_error(y, predictions)

        return mse, mae
