from object_detection.utils import label_map_util
import tensorflow as tf
#from common_imports import *
# Loads the graph
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
# ---------- LOAD THE CAR GOOGLE NET THAT IS A NUMPY  ----

import sys
sys.path.append('/home/watopedia/github_projects/model-convertion-repos/caffe-tensorflow/caffe-models/')
sys.path.append('/home/watopedia/github_projects/model-convertion-repos/caffe-tensorflow/')
from car_google_net import CarGoogleNet
def load_numpy_net_with_sess(PATH_TO_NPY, IMG_SIZE):
    '''
    Loads the numpy network in it's own session and returns it.
    '''
    batch_size = 1
    tf_image = tf.placeholder(tf.float32, [batch_size, IMG_SIZE, IMG_SIZE, 3])
    # Create an instance, passing in the input data
    net = CarGoogleNet({'data':tf_image})
    sess = tf.Session()

    net.load(PATH_TO_NPY,sess)
    return [net, sess, tf_image]

# Loading label map
def get_label_map(PATH_TO_LABELS):
    '''
    Returns the LABEL MAP and NUMBER of classes.
    '''
    if PATH_TO_LABELS is None:
        print('helper_nn_models_utils. get_label_map(). Warning. No Label Map Defined')
        return [None, None]
    NUM_CLASSES = 500
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    num_classes = len(category_index)
    return [category_index, num_classes]
