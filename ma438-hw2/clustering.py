import math
import sys
import pandas as pd
import numpy as np
from cluster import cluster

datasetPath = ""
K = 0
model = ""
xLen = 0
X = []


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



def printCentroid(k, latitude, longitude, reviewCount, checkins):
    print "Centroid%d=[%f,%f,%f,%f]" % (k, latitude, longitude, reviewCount, checkins)


# c: cluster 1, cluster 2
def cluster_distance(c1, c2):
    result = 0
    for x1 in c1.data:
        for x2 in c2.data:
            result += dist(x1, x2)

    return result / (c1.data.shape[0] * c2.data.shape[0])


# cs: multiple clusters, cs's len is K
# within-cluster sum of squared errors
def wc(cs,data):
    result = 0
    for c in cs:
        for x in c.data:
            result += dist(X[x], c.center) ** 2

    return result


def kMean(data):
    # Step 1 - Pick K random points as cluster centers called centroids.
    cs = []
    idx = np.random.randint(data.shape[0], size=K)
    centers = np.take(X, idx, axis=0)
    count = 0
    for center in centers:
        print center
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

        print "Score" + str(wc(cs,data))

        # Step 4 - break if certain iteration met
        count += 1
        if count == 10000: break


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
