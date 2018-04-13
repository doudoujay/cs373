import pandas as pd
import sys
from sklearn.model_selection import train_test_split

def split_normal(file):
    # No need to get dummy. Created for nbc
    f = pd.read_csv(file, sep=',', quotechar='"', header=0)
    train, test = train_test_split(f, test_size=0.2)
    train.to_csv("train.csv", quotechar='"', header=True)
    test.to_csv("test.csv", quotechar='"', header=True)
    
def split(file):
    f = pd.read_csv(file, sep=',', quotechar='"', header=0)
    # print f
    # preprocess
    f = pd.get_dummies(f, columns=f.columns.values)
    train, test = train_test_split(f, test_size=0.2)
    train.to_csv("train.csv", quotechar='"', header=True)
    test.to_csv("test.csv", quotechar='"', header=True)
if __name__ == "__main__":
    if sys.argv[2] == "normal":
        split_normal(sys.argv[1])
    else:    
        split(sys.argv[1])