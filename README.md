# Deep SORT with Tensorflow

## Introduction

This repository is an implementation to perform realtime tracking with Tensorflow using a [SSD model trained on the COCO dataset](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). It is based on the *Simple Online and Realtime Tracking with a Deep Association Metric* [Deep SORT](https://github.com/nwojke/deep_sort) algorithm. See the original repository for more information.

![alt text](https://github.com/omarabid59/TensorflowDeepSortTracking/blob/master/output_9Diy2e.gif)

## Dependencies
It's recommended that this is run in a python virtual environment [see here for more information](https://docs.python.org/3/library/venv.html). Ensure all of the dependencies in the [Deep SORT](https://github.com/nwojke/deep_sort) are installed.

Then install the dependencies with:
- `pip3 install -r requirements.txt`


## Setup
1. Download the [SSD Model](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
2. Copy the ```frozen_inference_graph.pb``` to the root directory of this repository.
3. Download the [Label Map](https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt)
4. Copy ```mscoco_label_map.pbtxt``` that you just downloaded to the root directory of this repository.

Your directory structure should look something like this:
```
  ObjectTracking/
  threads/
  utilities/
  README.md
  object_tracking.py
  frozen_inference_graph.pb
  mscoco_label_map.pbtxt
```

## Basic Usage
Run the file in your terminal by typing in ```python object_tracking.py```. The script will open up your webcam feed and begin detecting and tracking. The bounding boxes with the class labels are the result of detection from the SSD model. The overlayed blue bounding boxes are the output of the DeepSORT tracker.

If everything goes well, the system should be tracking in real time. Simply press ```Q``` to exit.

**Update: September 7, 2019**
- As requested by some individuals, I've added an option to use video input instead of the webcam. Do so by typing `python object_tracking.py --input VIDEO_FILE.mp4`. By default, the video is set to constantly loop through. See the `threads/ImageInput/VideoThread.py` file for implementation.
- Removed the Tensorflow Research [See here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) dependencies. Instead, the file required from this repository is copied and can be found at `utilities/external/visualization.py`. I do not take credit for this file!

## Issues
No issues found thus far, but please report any.
