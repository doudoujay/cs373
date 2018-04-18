import numpy as np
import pandas as pd


def zero_one(y, y_hat):
    # y and y_hat same dim
    n = y.shape[0]
    temp = 0
    for i in range(0, n):
        temp += 0 if y[i] == y_hat[i] else 1
    return temp/float(n)


def squared(y, p):
    n = y.shape[0]
    temp = 0
    for i in range(0, n):
        temp += (1-p[i]) ** 2
    return temp/float(n)


def baselinea_zero_one(y_train, y_test):
    n = y_test.shape[0]
    dictY = y_train.value_counts().to_dict()
    target = 0 if dictY[0] > dictY[1] else 1
    y_hat = np.zeros(n)
    y_hat.fill(target)
    return zero_one(y_test, y_hat)
