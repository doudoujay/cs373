# Implement a binary decision
# Implement a binary decision tree with a given maximum depth.
import os
import sys
import pandas as pd
import numpy as np
import scipy.stats
from Tree import Tree

data = None
X = None
trainPercentage = 0.0


def main(datasetPath, model, percentage):
    global data
    global X
    global trainPercentage
    # save train percent
    trainPercentage = int(percentage) / 100.0

    data = pd.read_csv(datasetPath, sep=',', quotechar='"', header=0)
    features = ['stars', 'priceRange', 'goodForKids', 'caters',
                'outdoorSeating', 'waiterService', 'attire', 'noiseLevel',
                'alcohol', 'open', 'goodForGroups']
    data = data[features]
    X = data.as_matrix()
    X = X[:1000]  # debug
    data = data.sample(frac=1)  # shuffle the dataset for train and test
    data = data.iloc[:1000,:]
    if model == "vanilla":
        vanilla()
    elif model == "depth":
        depth()
    elif model == "prune":
        prune()


def id3(sub_data, label):
    if check_same_label(sub_data):
        # If all examples are have same label
        node = Tree()
        node.data = sub_data
        node.label = label
        return node
    else:
        node = Tree()
        node.attr = best_attribute(sub_data)
        for idx, label in enumerate(sub_data['node.attr'].value_counts.keys()):
            if idx == 0 :
                node.left = id3(sub_data.loc[sub_data[node.attr] == label], label)
            if idx == 1:
                node.right = id3(sub_data.loc[sub_data[node.attr] == label], label)
        return node
        

def vanilla():
    """
    the full decision tree
    """
    # splitting to training and test
    trainSize = int(trainPercentage * len(data))
    training, test = data.iloc[:trainSize, :], data.iloc[trainSize:, :]
    print best_attribute(training)
    print_all_labels(training)
    return 0



def depth():
    """
    the decision tree with static depth
    """
    depthVal = int(sys.argv[5])
    # splitting to training, test and validation
    trainSize = int(trainPercentage * len(data))
    validSize = int(int(sys.argv[4])/100.0 * len(data))

    training, test = X[:trainSize, :], X[trainSize:, :]
    valid, test = test[:validSize, :],  test[validSize:, :]
    # print len(training)
    # print len(valid)
    # print len(test)
    return 0


def prune():
    """
    the decision tree with post-pruning
    """

    return 0


def H(S):
    """
    entropy
    """

    p_data = S.value_counts()/len(data)
    entropy = scipy.stats.entropy(p_data)
    return entropy


def IG(data, A):
    """
    information gainz
    """
 
    # print data.loc[data['open'] == True]
    temp = 0
    for k in A.value_counts().keys():
        sub_data = data.loc[A == k]
        weight = len(sub_data) / float(len(data))
        temp += weight * H(sub_data['goodForGroups'])
    # print H(data['goodForGroups'])
    # print temp
    return H(data['goodForGroups']) - temp

def check_same_label(data):
    return len(data['goodForGroups'].value_counts().keys()) == 1

def best_attribute(data):
    IG_arr = []
    for col in list(data):
        if col == 'goodForGroups': continue
        IG_arr.append(IG(data, data[col]))
    maxIdx = IG_arr.index(max(IG_arr))
    return list(data)[maxIdx]

def print_all_labels(data):
    for col in list(data):
        if col == 'goodForGroups': continue
        print data[col].value_counts().keys()
        


if __name__ == "__main__":
    if(len(sys.argv) < 4):
        print "Invalid input"
        exit
    main(sys.argv[1], sys.argv[2], sys.argv[3])
