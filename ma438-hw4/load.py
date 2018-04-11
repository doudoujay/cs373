import pandas as pd
def load(train_set, test_set):
    train = pd.read_csv(train_set,sep=',' , quotechar='"', header=0)
    test = pd.read_csv(test_set,sep=',' , quotechar='"', header=0)

    # drop redundent index
    train = train.drop(train.columns[[0]], axis=1)
    test = test.drop(test.columns[[0]], axis=1)
    # print train
    s = 307
    X_train, Y_train = train.iloc[:, :s], train.iloc[:, s:]
    X_test, Y_test = test.iloc[:, :s], test.iloc[:, s:]
    
    return (X_train, X_test, Y_train, Y_test)

