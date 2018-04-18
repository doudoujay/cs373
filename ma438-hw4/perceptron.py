import numpy as np
import pandas as pd
from loss import zero_one


class perceptron:
    def __init__(self, iter, X, Y, lr=1):
        self.X = X
        self.Y = Y
        self.iter = iter
        self.lr = lr
        return

    def train(self):
        # init weights
        self.w = np.zeros((self.X.shape[1]))
        # init bias
        self.b = 0
        for i in range(0, self.iter):
            for entry in range(0, self.X.shape[0]):
                error = self.Y[entry] - self.activation(self.X.loc[entry])
                if error != 0:
                    # update bias
                    self.b += error
                    # update weights
                    self.w += error * self.X.loc[entry] * self.lr
        return self.w, self.b

    def test(self, X_test, Y_test):
        y = Y_test
        y_hat = []
        for i in range(0, X_test.shape[0]):
            row = X_test.loc[i]
            y_hat.append(self.predict(row))
        res = zero_one(y, y_hat)
        # print res
        return res

    def activation(self, row):
        temp = np.dot(row, self.w.T) + self.b
        return 1 if temp >= 0 else 0

    def predict(self, row):
        temp = np.dot(row, self.w.T) + self.b
        return 1 if temp >= 0 else 0


class perceptronAverage(perceptron):
    def train(self):
        # init weights
        self.w = np.zeros((self.X.shape[1]))
        self.a = np.zeros((self.X.shape[1]))
        # init bias
        self.b = 0
        self.b_cahced = 0
        n = self.X.shape[0]
        step = n * self.iter
        for i in range(0, self.iter):
            for entry in range(0, self.X.shape[0]):
                error = self.Y[entry] - self.activation(self.X.loc[entry])
                if error != 0:
                    # update bias
                    self.b += error
                    # update weights
                    self.w += error * self.X.loc[entry] * self.lr
                    # update a
                    self.a += (float(step) / (n * self.iter)) * error * self.X.loc[entry] * self.lr
                    self.b_cahced += (float(step) / (n * self.iter)) * error
                step -= 1
        # print self.a
        return self.a, self.b_cahced

    def predict(self, row):
        temp = np.dot(row, self.a.T) + self.b_cahced
        return 1 if temp >= 0 else 0
