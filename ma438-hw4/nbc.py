import pandas as pd
import numpy as np
from util import load_normal
import sys
import math
from loss import *
from split import *

class NaiveBayes():
    def __init__(self, X_train, Y_train):
        self.X = X_train
        self.Y = Y_train
        self.data = pd.concat([X_train, Y_train], axis=1)
        self.summary_one = {}
        self.summary_zero = {}
        self.calcSummary()
        # print self.summary_zero
        # print self.summary_one
        self.zeroCnt = self.Y.value_counts().to_dict()[0]
        self.oneCnt = self.Y.value_counts().to_dict()[1]
        # prior
        self.pZero = self.zeroCnt / float(self.zeroCnt+self.oneCnt)
        self.pOne = self.oneCnt / float(self.zeroCnt+self.oneCnt)

    def calcSummary(self):
        self.X_zero = self.data.loc[self.data.goodForGroups == 0].iloc[:, :14]
        self.X_one = self.data.loc[self.data.goodForGroups == 1].iloc[:, :14]

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
                        self.summary_zero[k] = self.prob(n, n_c, len(labelCnt))

            elif val == 1:
                # put in summary one
                n = len(self.X_one)
                for key in self.X_one.keys():
                    col = self.X_one[key]
                    labelCnt = col.value_counts().to_dict()
                    for label, n_c in labelCnt.iteritems():
                        k = str(key) + " " + str(label)
                        self.summary_one[k] = self.prob(n, n_c, len(labelCnt))

    def predict(self, row):
        # TODO
        # zero
        zero = self.pZero
        # one
        one = self.pOne
        for key, label in row.iteritems():
            k = str(key) + " " + str(label)
            if k not in self.summary_zero:
                n = len(self.X_zero)
                col = self.X_zero[key]
                zero *= self.prob(n, 0, len(col.value_counts().to_dict()))
            else:
                zero *= self.summary_zero[k]
            if k not in self.summary_one:
                n = len(self.X_one)
                col = self.X_one[key]
                one *= self.prob(n, 0, len(col.value_counts().to_dict()))
            else:
                one *= self.summary_one[k]
        # print zero, one
        trueClassProb = one / float(zero+one)
        return (1, trueClassProb) if one > zero else (0, trueClassProb)

    def test_zero_one(self, X_test, Y_test):
        #
        y_hat = []
        y = Y_test
        for i in range(0, len(y)):
            y_hat.append(self.predict(X_test.iloc[i])[0])
        return zero_one(y, y_hat)

    def test_square(self, X_test, Y_test):
        p = []
        y = Y_test
        for i in range(0, len(y)):
            # print self.predict(X_test.iloc[i])
            p.append(self.predict(X_test.iloc[i])[1])
        return squared(y, p)

    def prob(self, n, n_c, k):
        if n_c != 0:
            return float(n_c) / float(n)
        return float(n_c+1) / float(n + k)

    @staticmethod
    def pdf(x, mean, sd):
        var = float(sd) ** 2
        pi = 3.1415926
        denom = (2 * pi * var) ** .5
        num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        return num / denom

    def calcProb(self, attribue):
        # For question in 2
        targetLabel = self.Y.unique()
        for i, val in enumerate(targetLabel):
            if val == 0:
                n = len(self.X_zero)
                for key in self.X_zero.keys():
                    if key == attribue:
                        col = self.X_zero[key]
                        labelCnt = col.value_counts().to_dict()
                        for label, n_c in labelCnt.iteritems():
                            k = str(key) + " " + str(label) + "goodForGroup = " + str(val)
                            print k, self.prob(n, n_c, len(labelCnt))

            elif val == 1:
                # put in summary one
                n = len(self.X_one)
                for key in self.X_one.keys():
                    if key == attribue:
                        col = self.X_one[key]
                        labelCnt = col.value_counts().to_dict()
                        for label, n_c in labelCnt.iteritems():
                            k = str(key) + " " + str(label) + "goodForGroup = " + str(val)
                            print k, self.prob(n, n_c, len(labelCnt))


def main(train, test):
    split_normal("yelp_cat.csv")
    X_train, X_test, Y_train, Y_test = load_normal(train, test)
    nb = NaiveBayes(X_train, Y_train)
    print "ZERO-ONE LOSS=" + str(nb.test_zero_one(X_test, Y_test))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
