import os
# Root Directory
ROOT_DIR = os.path.realpath(__file__)[:-22]
# Model Constants
CKPT_COCO = ROOT_DIR + 'frozen_inference_graph.pb'
LABELS_COCO = ROOT_DIR + 'mscoco_label_map.pbtxt'
