import numpy as np
import pandas as pd
def zero_one(y, y_hat):
    # y and y_hat same dim
    n = y.shape[0]
    temp = 0
    for i in range(0, n):
        temp += 0 if y[i] == y_hat[i] else 1
    return temp/float(n)
def squared():
    return