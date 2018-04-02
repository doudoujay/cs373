
class Tree(object):
    """Tree data structure
    left: left Tree
    right: right Tree
    data: sub_data for each layer
    """

    def __init__(self):

        self.left = None
        self.right = None
        self.data = None
        self.validData = None
        self.attr = ""
        self.leftLabel = ""
        self.rightLabel = ""
        # Calculated by valid data
        self.total = 0 # Total Len of data
        self.pos = 0 # Pos prediction if <=50K
        self.label = "" # label
    def isLeaf(self):
        return (self.left is None) and (self.right is None)
    def printTree(self):
        return 0
    def updatePruningData(self):
        self.total = len(self.validData)
        self.pos = len(self.validData.loc[self.validData['salaryLevel'] == " <=50K"])
        if self.pos > (self.total - self.pos):
            self.label = " <=50K"
        else:
            self.label = " >50K"


def getfullCount(root):
    # Base Case
    if root is None:
        return 0

    # Create an empty queue for level order traversal
    queue = []

    # Enqueue Root and initialize count
    queue.append(root)

    count = 0  # initialize count for full nodes
    while (len(queue) > 0):
        node = queue.pop(0)

        # if it is full node then increment count
        if node.left is not None and node.right is not None:
            count = count + 1

        # Enqueue left child
        if node.left is not None:
            queue.append(node.left)

        # Enqueue right child
        if node.right is not None:
            queue.append(node.right)

    return count