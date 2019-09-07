import threading
from abc import ABC, abstractmethod
import tensorflow as tf
import cv2


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
        try:
            self.category_index, self.NUM_CLASSES = self.get_label_map(
                PATH_TO_LABELS)
        except:
            self.done = True
            self.category_index = None
            self.NUM_CLASSES = None
            print("Error. Unable to load label map. Check your paths!")

    def get_label_map(self, labels_path):
        with open(labels_path, "r") as f:
            file_content = f.read()

        data = file_content.split('item')
        output_data = {}
        num_classes = 0
        for indx, x in enumerate(data):
            if len(x) == 0:
                continue
            name = x.split('name:')[1].split('\n')[0].strip().replace('"','')
            _id = x.split('id:')[1].split('\n')[0].strip()
            display_name = x.split('display_name:')[1].split('\n')[0].strip().replace('"','')
            output_data[indx] = {
                'name':display_name,
                'id':_id,
                'display_name':name
            }
            num_classes += 1
        category_index = output_data
        return category_index, num_classes

    def load_model(self, PATH_TO_MODEL):
        print('Loading Model File: ' + PATH_TO_MODEL)

        def get_graph(PATH_TO_CKPT):
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.compat.v1.GraphDef()
                with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                    return detection_graph

        def load_graph_with_sess(PATH_TO_CKPT):
            '''
            Loads the graph in it's own session and returns it.
            '''
            graph = get_graph(PATH_TO_CKPT)
            sess = tf.compat.v1.Session(graph=graph)
            return graph, sess
        # Load the graphs.
        [graph, sess] = load_graph_with_sess(PATH_TO_MODEL)
        print('Finished Loading Model')
        return graph, sess

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
    def predict(self, threadName):
        pass

    @abstractmethod
    def predict_once(self, image_np):
        pass

    def getImage(self):
        '''
        Returns the resized image that we will use for prediction.
        '''
        if self.IMG_SCALE < 1.0:
            self.output_data.image_np = cv2.resize(
                self.image_data.image_np.copy(), (0, 0),
                fx=self.IMG_SCALE, fy=self.IMG_SCALE)
        else:
            self.output_data.image_np = self.image_data.image_np
        return self.output_data.image_np
