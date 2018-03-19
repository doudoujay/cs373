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
        self.attr = ""
        self.leftLabel = ""
        self.rightLabel = ""
    def isLeaf(self):
        return (self.left is None) and (self.right is None)