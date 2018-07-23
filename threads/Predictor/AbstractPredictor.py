import threading
from abc import ABC,  abstractmethod

import tensorflow as tf
import numpy as np

from object_detection.utils import label_map_util



class AbstractPredictor(threading.Thread, ABC):
    def __init__(self, name, PATH_TO_CKPT,
                 PATH_TO_LABELS,
                 IMG_SCALE,
                 WITH_TRACKER=True,
                 ENABLE_BY_DEFAULT=False):
      
        threading.Thread.__init__(self)
        ABC.__init__(self)
        self.threadName = name
        self.done = False
        self.pause = not ENABLE_BY_DEFAULT
        
        self.IMG_SCALE = IMG_SCALE
        
        # TRACKER
        self.WITH_TRACKER = WITH_TRACKER
        

        
        [self.category_index, self.NUM_CLASSES] = self.get_label_map(PATH_TO_LABELS)
        
    def get_label_map(self, PATH_TO_LABELS):
        '''
        Returns the LABEL MAP and NUMBER of classes.
        '''
        if PATH_TO_LABELS is None:
            print('get_label_map(). Warning. No Label Map Defined')
            return [None, None]
        NUM_CLASSES = 500
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        num_classes = len(category_index)
        return [category_index, num_classes]
        
    def load_model(self, PATH_TO_MODEL):
            print('Loading Model File: ' + PATH_TO_MODEL)
            def get_graph(PATH_TO_CKPT):
                detection_graph = tf.Graph()
                with detection_graph.as_default():
                    od_graph_def = tf.GraphDef()
                    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                        serialized_graph = fid.read()
                        od_graph_def.ParseFromString(serialized_graph)
                        tf.import_graph_def(od_graph_def, name='')
                        return detection_graph
            def load_graph_with_sess(PATH_TO_CKPT):
                '''
                Loads the graph in it's own session and returns it.
                '''
                graph = get_graph(PATH_TO_CKPT)
                sess = tf.Session(graph=graph)
                return graph, sess
            # Load the graphs.
            [graph, sess] = load_graph_with_sess(PATH_TO_MODEL)
            print('Finished Loading Model')
            return [graph, sess]
    
    def run(self):
        print("Starting " + self.threadName)
        self.predict(self.threadName)
        print("Exiting " + self.threadName)
    def pause_predictor(self):
        self.pause = True
    def continue_predictor(self):
        self.pause = False
    def stop(self):
        self.done = True

    @abstractmethod
    def predict(self,threadName):
        pass
    @abstractmethod
    def predict_once(self,image_np):
        pass
    
    def getImage(self):
        '''
        Returns the resized image that we will use for prediction.
        '''
        if self.IMG_SCALE < 1.0:
            self.output_data.image_np =  cv2.resize(self.image_data.image_np.copy(), (0,0), fx=self.IMG_SCALE, fy=self.IMG_SCALE) 
        else:
            self.output_data.image_np = self.image_data.image_np
        return self.output_data.image_np
    
