import pandas as pd
import numpy as np
from util import load_normal
import sys
import math
from loss import *

class NaiveBayes():
    def __init__(self, X_train, Y_train):
        self.X = X_train
        self.Y = Y_train
        self.data = pd.concat([X_train, Y_train], axis=1)
        self.summary_one = {}
        self.summary_zero = {}
        self.calcSummary()
        print self.summary_zero
        print self.summary_one

    def train(self):
        return

    def test(self):
        return

    def calcSummary(self):
        self.X_zero = self.data.loc[self.data.goodForGroups == 0].iloc[:, :14]
        self.X_one = self.data.loc[self.data.goodForGroups == 1].iloc[:, :14]
        p = 0.5  # prior
        m = 3
        targetLabel = self.Y.unique()
        for i, val in enumerate(targetLabel):
            if val == 0:
                # put in summary_zero
                n = len(self.X_zero)
                for key in self.X_zero.keys():
                    col = self.X_zero[key]
                    labelCnt = col.value_counts().to_dict()
                    for label, n_c in labelCnt.iteritems():
                        k = str(key) + " " + str(label)
                        self.summary_zero[k] = self.prob(n, n_c, p, m)

            elif val == 1:
                # put in summary one
                n = len(self.X_one)
                for key in self.X_one.keys():
                    col = self.X_one[key]
                    labelCnt = col.value_counts().to_dict()
                    for label, n_c in labelCnt.iteritems():
                        k = str(key) + " " + str(label)
                        self.summary_one[k] = self.prob(n, n_c, p, m)

    def predict(self, row):
        p = 0.5
        m = 3
        # zero
        zero = 1
        # one
        one = 1
        for key, label in row.iteritems():
            k = str(key) + " " + str(label)
            if k not in self.summary_zero:
                n = len(self.X_zero)
                zero  *= self.prob(n, 0, p, m)
            else:
                zero *= self.summary_zero[k]
            if k not in self.summary_one:
                n = len(self.X_one)
                one *= self.prob(n, 0, p, m)
            else:
                one *= self.summary_one[k]
        # print zero, one
        return (1, one) if one > zero else (0, zero)

    def test(self, X_test, Y_test):
        temp = 0
        y = Y_test
        for i in range(0,len(y)):
            if self.predict(X_test.iloc[i])[0] == y.iloc[i]:
                temp += 1
        return temp/float(len(Y_test))

    def prob(self, n, n_c, p, m):
        return float(n_c + m * p) / float(n + m)

    @staticmethod
    def pdf(x, mean, sd):
        var = float(sd) ** 2
        pi = 3.1415926
        denom = (2 * pi * var) ** .5
        num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        return num / denom


def main(train, test):
    X_train, X_test, Y_train, Y_test = load_normal(train, test)
    nb = NaiveBayes(X_train, Y_train)
    print nb.test(X_test,Y_test)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
