import pandas as pd
def load(train_set, test_set, s=307):
    train = pd.read_csv(train_set,sep=',' , quotechar='"', header=0)
    test = pd.read_csv(test_set,sep=',' , quotechar='"', header=0)

    # drop redundent index
    train = train.drop(train.columns[[0]], axis=1)
    test = test.drop(test.columns[[0]], axis=1)
    # print train
    X_train, Y_train = train.iloc[:, :s], train.iloc[:, s+1]
    X_test, Y_test = test.iloc[:, :s], test.iloc[:, s+1]
    
    return (X_train, X_test, Y_train, Y_test)

def load_normal(train_set, test_set):
    return load(train_set, test_set, s=14)