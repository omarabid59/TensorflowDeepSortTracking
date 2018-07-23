import numpy as np
class OutputNNData:
    def __init__(self):
        self.scores = np.asarray([])
        self.classes = np.asarray([])
        self.image_np = np.asarray([])
        self.category_index = ()
class OutputCarMakeModelData:
    def __init__(self):
        self.scoresPerClass = ()
        #self.image_np = ()
class OutputClassificationData(OutputNNData):
    def __init__(self):
        super().__init__()
        self.bbs = np.asarray([])
        self.score_thresh = ()
class OutputVisData:
    def __init__(self):
        self.visualization_data = ()
        
class ImageData:
    def __init__(self):
        self.image_np = ()
class OutputFaceData(OutputNNData):
    def __init__(self):
        super().__init__()
        self.persons = np.asarray([])
        self.bbs = np.asarray([])
