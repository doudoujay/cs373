import pandas as pd
import numpy as np
from load import load
import sys
def main(train_set, test_set, iter=2):
    X_train, X_test, Y_train, Y_test = load(train_set, test_set)
    print Y_train


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])