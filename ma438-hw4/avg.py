from perceptron import perceptronAverage
from util import load
import sys

def main(train_set, test_set, iter=2):
    iter = int(iter)
    X_train, X_test, Y_train, Y_test = load(train_set, test_set)
    p = perceptronAverage(iter, X_train, Y_train)
    p.train()
    p.test(X_test,Y_test)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])