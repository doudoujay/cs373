import pandas as pd
import numpy as np
from perceptron import perceptron
from util import load
import sys
from split import * 

def main(train_set, test_set, iter=2):
    split("yelp_cat.csv")
    iter = int(iter)
    X_train, X_test, Y_train, Y_test = load(train_set, test_set)
    p = perceptron(iter, X_train, Y_train)
    p.train()
    print "ZERO-ONE LOSS=" + str(p.test(X_test,Y_test))


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])
