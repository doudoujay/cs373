# Implement a binary decision
# Implement a binary decision tree with a given maximum depth.
import os
import sys
import pandas as pd
import numpy as np
import scipy.stats
import Tree

data = None
X = None
trainPercentage = 0.0
def main(datasetPath, model, percentage):
    global data
    global X
    global trainPercentage
    #save train percent
    trainPercentage = int(percentage) / 100.0

    data = pd.read_csv(datasetPath, sep=',', quotechar='"', header=0)
    features = ['stars', 'priceRange', 'goodForKids', 'caters',
                'outdoorSeating', 'waiterService', 'attire', 'noiseLevel',
                'alcohol', 'open', 'goodForGroups']
    data = data[features]
    X = data.as_matrix()
    X = X[:1000] # debug
    np.random.shuffle(X) # shuffle the dataset for train and test
    if model == "vanilla":
        vanilla()
    elif model == "depth":
        depth()
    elif model == "prune":
        prune()
# the full decision tree
def vanilla():
    #splitting to training and test
    trainSize = int(trainPercentage * len(X))
    training, test = X[:trainSize,:], X[trainSize:,:]

    IG(data['open'],data['outdoorSeating'])


    return 0
# the decision tree with static depth
def depth():
    depthVal = int(sys.argv[5])
    #splitting to training, test and validation
    trainSize = int(trainPercentage * len(X))
    validSize = int(int(sys.argv[4])/100.0 * len(X))
    training, test = X[:trainSize,:], X[trainSize:,:]
    valid, test = test[:validSize, :],  test[validSize:, :]
    # print len(training)
    # print len(valid)
    # print len(test)
    return 0

# the decision tree with post-pruning
def prune():
    return 0

# information gainz
# How much does a feature split decrease the entropy
def IG(S, A):
    print S.value_counts()
    print A.value_counts()
    temp = 0
    for v in A.value_counts().values:
        temp += (v / sum(A.value_counts().values) *

    return H(S) - temp


# entropy
def H(S):
    p_data= S.value_counts()/len(data)
    entropy=scipy.stats.entropy(p_data)
    return entropy

if __name__ == "__main__":
    if(len(sys.argv) < 4):
        print "Invalid input"
        exit
    main(sys.argv[1], sys.argv[2], sys.argv[3])
