import math
import sys
import pandas as pd
import numpy as np
from cluster import cluster
from distHeap import distHeap
from sklearn import preprocessing

datasetPath = ""
K = 0
model = ""
xLen = 0
X = []
globalSse = 0
globalCs=[]

def main(d, k, m):
    global datasetPath
    global xLen
    global K
    global model
    global X

    datasetPath = d
    K = int(k)
    model = m
    data = pd.read_csv(datasetPath, sep=',', quotechar='"', header=0)
    data = data[['latitude', 'longitude', 'reviewCount', 'checkins']]
    X = data.as_matrix()
    xLen = len(X[0])
    # for x in X:
    #     print x
    if model == 'km':
        kMean(data)
    elif model == 'ac':
        agglomerative(data)

# question 2.c
def logMain(d,k,m):
    global datasetPath
    global xLen
    global K
    global model
    global X

    datasetPath = d
    K = int(k)
    model = m
    data = pd.read_csv(datasetPath, sep=',', quotechar='"', header=0)
    data = data[['latitude', 'longitude', 'reviewCount', 'checkins']]
    data['reviewCount'] = np.log(data['reviewCount'])
    data['checkins'] = np.log(data['checkins'])
    X = data.as_matrix()
    xLen = len(X[0])
    # for x in X:
    #     print x
    if model == 'km':
        kMean(data)
    elif model == 'ac':
        agglomerative(data)

def skMain(d,k,m):
    global datasetPath
    global xLen
    global K
    global model
    global X

    datasetPath = d
    K = int(k)
    model = m
    data = pd.read_csv(datasetPath, sep=',', quotechar='"', header=0)
    data = data[['latitude', 'longitude', 'reviewCount', 'checkins']]
    data = preprocessing.scale(data)
    X = data
    xLen = len(X[0])
    # for x in X:
    #     print x
    if model == 'km':
        kMean(data)
    elif model == 'ac':
        agglomerative(data)



def printCentroid(k, latitude, longitude, reviewCount, checkins):
    print "Centroid%d=[%f,%f,%f,%f]" % (k, latitude, longitude, reviewCount, checkins)




# c: cluster 1, cluster 2
def cluster_distance(c1, c2):
    result = 0
    for x1 in c1.data:
        for x2 in c2.data:
            result += dist(X[x1], X[x2])

    return result / (len(c1.data) * len(c2.data))


# cs: multiple clusters, cs's len is K
# within-cluster sum of squared errors
def wc(cs,data):
    result = 0
    for c in cs:
        for x in c.data:
            result += dist(X[x], c.center) ** 2

    return result


def kMean(data):
    global globalSse
    global globalCs
    # Step 1 - Pick K random points as cluster centers called centroids.
    cs = []
    idx = np.random.randint(data.shape[0], size=K)
    centers = np.take(X, idx, axis=0)
    count = 0
    lastScore = 0
    for center in centers:
        cs.append(cluster(center))
    while True:
        # Step 2 - Assign each xi to nearest cluster by calculating its distance to each centroid.
        for c in cs:
            c.clearData()

        for idx, x in enumerate(X):
            cIndex = minDistCentroid(cs, x)
            cs[cIndex].data.append(idx)

        # Step 3 - Find new centroid by the new clusters
        for c in cs:
            newCentroid = np.mean(np.take(X, c.data, 0), axis=0)
            c.updateCenter(newCentroid)


        sse = str(wc(cs, data))
        # Step 4 - break if certain iteration met
        count += 1
        if count == 10000 or lastScore == sse:
            printResult(cs, sse)
            globalSse = sse
            globalCs = cs
            break
        lastScore = sse

# def agglomerative(data):
#     global globalSse
#     global globalCs
#     # first, every single point is a cluster
#     cs = []
#     checked = []
#     new_cs = []
#     for idx, x in enumerate(X):
#         c = cluster(x)
#         checked.append(False)
#         c.data.append(idx)
#         cs.append(c)
#     # iteratively pick two clusters where their distance is minimal and fuse them. The minumun distance will be decided by the average link cluster distance
#     while(True):
#         # break condition
#         print len(cs)
#         if len(cs) <= K:
#             sse = str(wc(cs, data))
#             printResult(cs, sse)
#             globalSse = sse
#             globalCs = cs
#             break
#         for i, x in enumerate(cs):
#             dists = {}
#             for j, y in enumerate(cs):
#                 if i != j and not checked[i] and not checked[j]:
#                     dists[cluster_distance(cs[i], cs[j])] = j
#             if len(dists.keys()) == 0: continue
#             minKey = np.argmin(dists.keys())
#             j = dists[dists.keys()[minKey]]
#             # merge the data for two sets
#             newData = np.concatenate((cs[i].data, cs[j].data))
#             # Append the new cluster into the new cs
#             c = cluster(np.mean(np.take(X, newData, 0),  axis=0))
#             c.data = newData
#             new_cs.append(c)
#             # Set them as checked
#             checked[i] = True
#             checked[j] = True
#         # reset the clusters
#         cs = new_cs
#         new_cs = []
#         # reset checked markers
#         checked = []
#         for idx,c in enumerate(cs):
#             # checked.append(False)

def agglomerative(data):
# first, every single point is a cluster
    global globalSse
    global globalCs
    cs = []
    heap = distHeap()
    for idx, x in enumerate(X):
        c = cluster(x)
        c.data.append(idx)
        cs.append(c)
    for i, x in enumerate(cs):
        for j in range(i+1, len(cs)):
            dist = cluster_distance(cs[i], cs[j])
            # print "dist" + str(dist) + " " + str(i) + " " + str(j)
            heap.add_clusters(cs[i], cs[j], dist)

    while True:
        if len(cs) <= K:
            sse = str(wc(cs, data))
            printResult(cs, sse)
            globalSse = sse
            globalCs = cs
            break
        c1, c2 = heap.min_dist_clusters()

        # merge c1 and c2
        newData = np.concatenate((c1.data, c2.data))
        # Append the new cluster into the new cs
        c = cluster(np.mean(np.take(X, newData, 0),  axis=0))
        c.data = newData
        # remove associate c1 and c2
        for c_old in cs:
            # print 'remove'
            heap.remove_cluster(c1, c_old)
            heap.remove_cluster(c_old, c1)
            heap.remove_cluster(c2, c_old)
            heap.remove_cluster(c_old, c2)
        cs.remove(c1)
        cs.remove(c2)
        # add new
        for c_ind in cs:
            # print 'add new dist'
            dist = cluster_distance(c, c_ind)
            heap.add_clusters(c, c_ind, dist)
        # print c1.data
        # print c2.data
        cs.append(c)
def printResult(cs, wc):
    print 'WC-SSE='+wc
    for idx, c in enumerate(cs):
        printCentroid(idx+1, c.center[0], c.center[1], c.center[2], c.center[3])

def dist(x1, x2):
    result = 0
    for i in range(0, xLen):
        result += (x1[i] - x2[i]) ** 2
    return math.sqrt(result)


def minDistCentroid(cs, x):
    result = []
    for c in cs:
        result.append(dist(c.center, x))
    return np.argmin(result)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
