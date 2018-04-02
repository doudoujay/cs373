# Implement a binary decision
# Implement a binary decision tree with a given maximum depth.
import os
import sys
import pandas as pd
import numpy as np
import scipy.stats
import random
from Tree import Tree
from Tree import getfullCount

data = None
X = None
trainPercentage = 0.0
test = None
nodeCount = 0
train_accu = 0.0
test_accu = 0.0
valid_accu = 0.0
depthMode = False
depthVal = 0


def main(datasetPath, testPath, model, percentage):
    global data
    global test
    global X
    global trainPercentage
    #  training set percentage
    trainPercentage = int(percentage) / 100.0

    test = pd.read_csv(testPath, sep=',', quotechar='"', header=None, engine='python',
                       names=["workclass", "education", "marital-status", "occupation", "relationship", "race",
                              "sex", "native-country", "salaryLevel"])
    data = pd.read_csv(datasetPath, sep=',', quotechar='"', header=None, engine='python',
                       names=["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                              "native-country", "salaryLevel"])
    # features = ['stars', 'priceRange', 'goodForKids', 'caters',
    #             'outdoorSeating', 'waiterService', 'attire', 'noiseLevel',
    #             'alcohol', 'open', 'salaryLevel']
    # data = data[features]
    X = data.as_matrix()
    # X = X[:1000]  # debug
    # data = data.sample(frac=1)  # shuffle the dataset for train and test
    # data = data.iloc[:1000, :]
    if model == "vanilla":
        vanilla()
    elif model == "depth":
        depth(sys.argv[5], sys.argv[6])
    elif model == "prune":
        prune(sys.argv[5])


def depthHelper(datasetPath, testPath, percentage, validPercentage, depthVal):
    global data
    global test
    global X
    global trainPercentage
    #  training set percentage
    trainPercentage = int(percentage) / 100.0

    test = pd.read_csv(testPath, sep=',', quotechar='"', header=None, engine='python',
                       names=["workclass", "education", "marital-status", "occupation", "relationship", "race",
                              "sex", "native-country", "salaryLevel"])
    data = pd.read_csv(datasetPath, sep=',', quotechar='"', header=None, engine='python',
                       names=["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                              "native-country", "salaryLevel"])
    X = data.as_matrix()

    depth(validPercentage, depthVal)


def pruneHelper(datasetPath, testPath, percentage, validPercentage):
    global data
    global test
    global X
    global trainPercentage
    #  training set percentage
    trainPercentage = int(percentage) / 100.0

    test = pd.read_csv(testPath, sep=',', quotechar='"', header=None, engine='python',
                       names=["workclass", "education", "marital-status", "occupation", "relationship", "race",
                              "sex", "native-country", "salaryLevel"])
    data = pd.read_csv(datasetPath, sep=',', quotechar='"', header=None, engine='python',
                       names=["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
                              "native-country", "salaryLevel"])
    X = data.as_matrix()

    prune(validPercentage)


def id3(sub_data, depth, validData=None):

    global nodeCount
    # print depth
    if validData is None:
        validData = data
    nodeCount += 1
    if check_same_label(sub_data) or (depthMode and depth == depthVal):
        # leaf
        node = Tree()
        node.data = sub_data
        node.validData = validData

        node.updatePruningData()
        return node
    else:
        node = Tree()
        node.validData = validData

        node.updatePruningData()
        node.attr = best_attribute(sub_data)
        if (node.attr == None):
            return None
        # print "Best attr for node " + node.attr
        maxIG, label = IG_binary(sub_data, sub_data[node.attr])
        if maxIG is None and label is None:
            print "error"

        node.leftLabel = str(label)
        node.rightLabel = "Not " + str(label)
        node.left = id3(sub_data.loc[sub_data[node.attr] == label], depth + 1,
                        validData.loc[validData[node.attr] == label])
        node.right = id3(sub_data.loc[sub_data[node.attr] != label], depth + 1,
                         validData.loc[validData[node.attr] != label])
        if (node.left is None and node.right is None):
            node.data = sub_data
    # print nodeCount
    return node


def vanilla():
    """
    the full decision tree
    """
    global train_accu
    global nodeCount
    global test_accu
    # splitting to training and test
    trainSize = int(trainPercentage * len(data))
    training = data.iloc[:trainSize, :]
    # print best_attribute(training)
    # print len(training)
    # print len(test)
    # print_all_labels(data)
    node = id3(training, 0)
    train_accu = accuracy(node, training)
    test_accu = accuracy(node, test)
    print "Training set accuracy: " + str(train_accu)
    print "Test set accuracy: " + str(test_accu)
    nodeCount = getfullCount(node)


def depth(validVal, depthLocalVal):
    """
    the decision tree with static depth
    """
    global depthMode
    global train_accu
    global test_accu
    global valid_accu
    global depthVal
    global nodeCount

    depthMode = True
    # validation set percentage.
    validationPercentage = int(validVal) / 100.0
    validSize = int(validationPercentage * len(data))

    # value of maximum depth
    depthVal = depthLocalVal

    # splitting to training, test and validation
    trainSize = int(trainPercentage * len(data))

    training, testT = data.iloc[:trainSize, :], data.iloc[trainSize:, :]
    valid, testT = testT.iloc[:validSize, :], testT.iloc[validSize:, :]
    # print len(training)
    # print len(valid)
    # print depthVal
    node = id3(training, 0)
    train_accu = accuracy(node, training)
    valid_accu = accuracy(node, valid)
    test_accu = accuracy(node, test)
    print "Training set accuracy: " + str(train_accu)
    print "Validation set accuracy: " + str(valid_accu)
    print "Test set accuracy: " + str(test_accu)
    depthMode = False
    nodeCount = getfullCount(node)
    return 0


def prune(validVal):
    """
    the decision tree with post-pruning
    """
    global train_accu
    global test_accu
    global valid_accu
    global depthVal
    global nodeCount
    # validation set percentage.
    validationPercentage = int(validVal) / 100.0
    validSize = int(validationPercentage * len(data))

    # splitting to training, test and validation
    trainSize = int(trainPercentage * len(data))

    training, testT = data.iloc[:trainSize, :], data.iloc[trainSize:, :]
    valid, testT = testT.iloc[:validSize, :], testT.iloc[validSize:, :]
    # prune_error(valid, training)
    # build tree
    node = id3(training, 0, validData=valid)
    # prune the tree
    pruneTree(node)

    train_accu = accuracy(node, training)
    test_accu = accuracy(node, test)
    print "Training set accuracy: " + str(train_accu)
    print "Test set accuracy: " + str(test_accu)
    nodeCount = getfullCount(node)

    return 0


def H(S):
    """
    entropy
    """

    p_data = S.value_counts() / len(data)
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
        temp += weight * H(sub_data['salaryLevel'])
    # print H(data['salaryLevel'])
    # print temp
    return H(data['salaryLevel']) - temp


def IG_binary(data, A):
    """
    information gainz max for binary tree
    """
    temp = []
    labels = A.value_counts().keys()
    for k in labels:
        sub_data_k = data.loc[A == k]
        sub_data_not_k = data.loc[A != k]
        weight_k = len(sub_data_k) / float(len(data))
        weight_not_k = len(sub_data_not_k) / float(len(data))
        sub = weight_k * H(sub_data_k['salaryLevel']) + weight_not_k * H(sub_data_not_k['salaryLevel'])
        temp.append(H(data['salaryLevel']) - sub)
    if all(i == 0.0 for i in temp):
        return None, None
    maxIdx = temp.index(max(temp))
    # print temp
    # print labels
    # print "Best label to split " + str(labels[maxIdx])
    return (temp[maxIdx], labels[maxIdx])


def check_same_label(data):
    # print data['salaryLevel'].value_counts()
    return len(data['salaryLevel'].value_counts().keys()) <= 1


def best_attribute(data):
    IG_arr = []
    for col in list(data):
        if col == 'salaryLevel': continue
        IG_arr.append(IG(data, data[col]))
    if all(i == 0.0 for i in IG_arr):
        return None
    maxIdx = IG_arr.index(max(IG_arr))
    return list(data)[maxIdx]


def print_all_labels(data):
    for col in list(data):
        if col == 'salaryLevel': continue
        print data[col].value_counts().keys()


def handle_more_than_two(attr):
    labels = data['node.attr'].count_values().keys()


def handle_numerical(attr):
    return 0


def accuracy(node, data):
    temp = 0
    for index, row in data.iterrows():
        if inference(node, row) == row['salaryLevel']:
            temp += 1
    return temp / float(len(data))


def inference(node, data_row):
    if node.isLeaf():
        values = node.data['salaryLevel'].value_counts().keys()
        return random.choice(values)
    dataLabel = data_row[node.attr]
    if dataLabel == node.leftLabel and node.left != None:
        # go to left node
        return inference(node.left, data_row)
    elif node.right != None:
        # go to right node
        return inference(node.right, data_row)


def pruneTree(node):
    if node is None: return 0
    if node.isLeaf():
        # leaf here
        if node.label == " <=50K":
            return node.total - node.pos
        else:
            return node.pos
    else:
        error = pruneTree(node.left) + pruneTree(node.right)
        if error < min(node.pos, (node.total - node.pos)):
            return error
        else:
            # replace with leaf
            # merge data
            # node.validData = (node.left.validData if node.left is not None else None) + (node.right.validData if node.right is not None else None)
            # node.updatePruningData()
            leftD = node.left.data if node.left is not None else None
            rightD = node.right.data if node.right is not None else None
            node.data = pd.concat([leftD, rightD], axis=0)
            node.left = None
            node.right = None

            if node.label == " <=50K":
                return node.total - node.pos
            else:
                return node.pos


def prune_error(validData, data):
    valuesData = data.groupby(['salaryLevel']).size()
    valueValidData = validData.groupby(['salaryLevel']).size()
    print valuesData
    print valueValidData
    # temp = 0
    # for index, row in data.iterrows():
    #     if inference(node, row) == row['salaryLevel']:
    #         temp += 1


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print "Invalid input"
        exit
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
