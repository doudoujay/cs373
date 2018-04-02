class cluster:
    # data should be array of indexs
    def __init__(self, center):
        self.center = center
        self.data = []
    def updateCenter(self, center):
        self.center = center
    def clearData(self):
        self.data = []
