from threads.Predictor.AbstractPredictor import AbstractPredictor
import numpy as np
import time


class OutputClassificationData():
    def __init__(self):
        """
        Class to hold data.
        """
        self.bbs = np.asarray([])
        self.score_thresh = ()
        self.scores = np.asarray([])
        self.classes = np.asarray([])
        self.image_np = np.asarray([])
        self.category_index = ()


class PredictorImage(AbstractPredictor):
    def __init__(self, name, PATH_TO_CKPT,
                 PATH_TO_LABELS,
                 image_data,
                 score_thresh,
                 IMG_SCALE=1.0,
                 WITH_TRACKER=True,
                 ENABLE_BY_DEFAULT=False):

        super().__init__(name, PATH_TO_CKPT,
                         PATH_TO_LABELS,
                         IMG_SCALE,
                         WITH_TRACKER,
                         ENABLE_BY_DEFAULT)

        self.image_data = image_data
        self.output_data = OutputClassificationData()
        self.output_data.score_thresh = score_thresh
        self.output_data.category_index = self.category_index
        self.score_thresh = score_thresh

        self.detection_graph, self.sess = self.load_model(PATH_TO_CKPT)

        [self.image_tensor, self.boxes_tensor,
         self.scores_tensor, self.classes_tensor,
         self.num_detections_tensor] = self.getTensors()

    def getTensors(self):
        '''
        Returns the tensors
        '''
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')
        # Each box represents a part of the image where a particular object was
        # detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')
        return [image_tensor, boxes, scores, classes, num_detections]

    def predict(self, threadName):
        while not self.done:
            image_np = self.getImage().copy()
            if not self.pause:
                self.predict_once(image_np)
            else:
                self.output_data.bbs = np.asarray([])
                time.sleep(2.0)  # Sleep for 2 seconds

    def predict_once(self, image_np):
        # Actual detection.
        with self.sess.as_default(), self.detection_graph.as_default():
            [boxes, scores, classes,
                temp_num_detections] = self.inferNN(image_np)
            if self.WITH_TRACKER:
                [boxes, scores, classes] = self.runTracker(
                    boxes, scores, classes, image_np.copy(), self.score_thresh)
            else:
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes)
                # Eliminate all values that do not meet the threshold.
                indices = [
                    i for i, x in enumerate(
                        scores > self.output_data.score_thresh) if x]
                scores = scores[indices]
                classes = classes[indices]
                boxes = boxes[indices]

            self.output_data.bbs = boxes
            self.output_data.scores = scores
            self.output_data.classes = classes
            # For now, we simply pass a reference to the image. We will use
            # this for predicting on the subset.
            self.output_data.img = image_np
            if self.output_data.bbs == ():
                self.output_data.bbs = np.asarray([])
            time.sleep(0.1)

    def runTracker(self, boxes, scores, classes, image_np, score_thresh):
        [temp_box, temp_score, temp_class,
         _] = self.global_tracker.pipeline(boxes,
                                           scores,
                                           classes,
                                           image_np,
                                           score_thresh=self.score_thresh)
        return [temp_box, temp_score, temp_class]

    def inferNN(self, image_np):
        '''
        Perform inference on the image
        '''
        # Expand dimensions since the model expects images to have shape: [1,
        # None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        [boxes,
         scores,
         classes,
         temp_num_detections] = self.sess.run([self.boxes_tensor,
                                               self.scores_tensor,
                                               self.classes_tensor,
                                               self.num_detections_tensor],
                                              feed_dict={
                                                         self.image_tensor:
                                                         image_np_expanded
                                                         }
                                              )
        return [boxes, scores, classes, temp_num_detections]
