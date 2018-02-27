import numpy as np
class distHeap:
    dists = [] # dists arranged in heap
    clustersMap = {} # clusters(c1, c2)-> dist
    distsMap = {}
    validCluster = {}
    REMOVED = '<removed>'

    def __init__(self):
        pass
    def add_clusters(self, c1, c2, dist):
        self.validCluster[c1] = True
        self.validCluster[c2] = True
        clusters = (c1, c2)
        if clusters in self.clustersMap:
            return
        self.clustersMap[clusters] = dist
        self.distsMap[dist] = clusters
        self.dists.append(dist)

    def remove_cluster(self, c1, c2):
        # clusters = (c1, c2)
        # if clusters in self.clustersMap:
        #     dist = self.clustersMap[clusters]
        #     self.clustersMap.pop(clusters)
        #     self.distsMap.pop(dist)
        #     self.dists.remove(dist)
        self.validCluster[c1] = False
        self.validCluster[c2] = False




    def min_dist_clusters(self):
        minDist = self.dists[np.argmin(self.dists)]
        print minDist
        c = self.distsMap[minDist]

        while(self.validCluster[c[0]] == False or self.validCluster[c[1]] == False):
            self.clustersMap.pop(c)
            self.distsMap.pop(minDist)
            self.dists.remove(minDist)

            minDist = self.dists[np.argmin(self.dists)]
            c = self.distsMap[minDist]

        return c
