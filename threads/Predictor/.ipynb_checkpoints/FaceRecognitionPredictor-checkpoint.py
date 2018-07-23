"""@author: omarabid59
   @author: Phanikumar
   Face recognition module implemented as a seperate class using OpenFace
"""
import dlib
import imutils
import cv2
from imutils import face_utils
import sys
import pickle
import time
import openface
import numpy as np
from sklearn.mixture import GMM



# NEW
import constants
from AbstractPredictor import AbstractPredictor
from helper_data_encapsulation import OutputFaceData
class FaceRecognitionPredictor(AbstractPredictor):
    '''
    # Setup Face Recognition Module
    baseDir = '/home/watopedia/github_projects/aipod-cnet/models/face_recognition/'
    FACE_SHAPE_PREDICTOR = baseDir + 'shape_predictor_68_face_landmarks.dat'
    FACE_SHAPE_DNN_MODEL = baseDir + 'nn4.small2.v1.t7'
    FACE_CLASSIFIER_MODEL = baseDir + 'classifier.pkl'
    thread_frm = FRM(FACE_SHAPE_PREDICTOR,
                    FACE_SHAPE_DNN_MODEL, 
                      FACE_CLASSIFIER_MODEL,
                      webcam_thread)
    '''

    def __init__(self, name, image_data):
        super().__init__(name, None,
                 None,
                 IMG_SCALE = 1.0,
                 WITH_TRACKER=False,
                 ENABLE_BY_DEFAULT=False)
        
        sys.path.append('/home/watopedia/torch/install/bin/th')
        
        self.__FACE_SHAPE_PREDICTOR = constants.FACE_SHAPE_PREDICTOR
        self.__FACE_SHAPE_DNN_MODEL = constants.FACE_SHAPE_DNN_MODEL
        self.__FACE_CLASSIFIER_MODEL = constants.FACE_CLASSIFIER_MODEL
        
        
        self.align = openface.AlignDlib(self.__FACE_SHAPE_PREDICTOR)
        self.net = openface.TorchNeuralNet(self.__FACE_SHAPE_DNN_MODEL,
                                           imgDim=96,
                                           cuda=False)
        
        # initialize dlib's face detector (HOG-based) and then create
        self.detector = dlib.get_frontal_face_detector()
        # the facial landmark predictor
        self.predictor = dlib.shape_predictor(self.__FACE_SHAPE_PREDICTOR)

        # Load the classification model
        [self.le, self.clf] = self.load_classifier_model(self.__FACE_CLASSIFIER_MODEL)
        
        # Initialize
        self.persons = []
        self.confidences = []
        self.bbs = []
        self.rects = []

        
        # Read the output data.
        self.output_data = OutputFaceData()
        self.class_index = []
        # Webcam Thread Reference
        self.image_data = image_data
        


    def bb_intersection_over_union(self, bboxes1, bboxes2):
        """
        Args:
            bboxes1: shape (total_bboxes1, 4)
                with x1, y1, x2, y2 point order.
            bboxes2: shape (total_bboxes2, 4)
                with x1, y1, x2, y2 point order.
            p1 *-----
               |     |
               |_____* p2
        Returns:
            Tensor with shape (total_bboxes1, total_bboxes2)
            with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
            in [i, j].
        """

        x11 = bboxes1.left()
        x12 = bboxes1.right()  
        y11 = bboxes1.bottom()
        y12 = bboxes1.top()  

        x21 = bboxes2.left()
        x22 = bboxes2.right()  
        y21 = bboxes2.bottom()
        y22 = bboxes2.top()     
        #x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
        #x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

        xI1 = max(x11, np.transpose(x21))
        yI1 = max(y11, np.transpose(y21))

        xI2 = min(x12, np.transpose(x22))
        yI2 = min(y12, np.transpose(y22))

        inter_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)

        bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
        bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

        union = (bboxes1_area + np.transpose(bboxes2_area)) - inter_area

        return max(inter_area / union, 0)
    
    def detect_faces(self, image):
    

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = self.detector(gray, 1)
        
        # This tries to match the bounding boxes of the two detectors
        # Then use the closest match to display the label here.
        self.max_overlap_indices = []
        self.max_overlap_values = []

        temp_bbs = list(self.bbs)
        for bb in rects:
            distances = []
            for box in temp_bbs:
                iou = self.bb_intersection_over_union(bb,box)
                distances.append(iou)
            if len(distances) > 0:
                self.max_overlap_indices.append(distances.index(max(distances)))
                self.max_overlap_values.append(max(distances))
            
        # okay. so, go through each rectangle outputted by the landmarks
        # IFF there is a corresponding bb , append it.
        # We should now also have the index value of the bbs.
        # That is directly associated
        
        self.rects = rects
        self.landmarks = []
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            self.landmarks.append(shape)            

    def draw_bb_and_landmarks(self, image):
        '''
        '''
        if self.pause:
            return image
        rects = list(self.rects)
        landmarks = list(self.landmarks)
        temp_persons = list(self.persons)
        max_overlap_indices = list(self.max_overlap_indices)
        for (i, rect) in enumerate(rects):
            
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if (i <= (len(max_overlap_indices) - 1) and len(max_overlap_indices) > 0):
                # Check to ensure there are as many persons as the name we want to display
                if len(temp_persons) >= len(max_overlap_indices):
                    cv2.putText(image, temp_persons[max_overlap_indices[i]], (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)





            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            shape = landmarks[i]
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 0), -1)
        return image
   
    def load_classifier_model(self, classifierModel):
        with open(classifierModel, 'rb') as f:
            if sys.version_info[0] < 3:
                    (le, clf) = pickle.load(f)  # le - label and clf - classifer
            else:
                    (le, clf) = pickle.load(f, encoding='latin1')  # le - label and clf - classifer
        return [le, clf]
    def getRep(self, bgrImg):
        if bgrImg is None:
            raise Exception("Unable to load image/frame")

        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

      

        # Get all bounding boxes
        bb = self.align.getAllFaceBoundingBoxes(rgbImg)

        if bb is None:
            # raise Exception("Unable to find a face: {}".format(imgPath))
            return None



        alignedFaces = []
        for box in bb:
            alignedFaces.append(
                self.align.align(
                    96,
                    rgbImg,
                    box,
                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

        if alignedFaces is None:
            raise Exception("Unable to align the frame")


        reps = []
        for alignedFace in alignedFaces:
            reps.append(self.net.forward(alignedFace))

        return [reps,bb]
    
    def predict_once(self, img):
        [reps, bbs] = self.getRep(img)

        persons = []
        confidences = []
        class_index = []
        for rep in reps:
            try:
                rep = rep.reshape(1, -1)
            except:
                print ("No Face detected")
                return (None, None)
            predictions = self.clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            class_index.append(maxI)
            persons.append(self.le.inverse_transform(maxI))
            confidences.append(predictions[maxI])
            if isinstance(self.clf, GMM):
                dist = np.linalg.norm(rep - self.clf.means_[maxI])
                pass
        return (class_index, persons, confidences ,bbs)
    
    def predict(self, threadName):
        # Since the coordinates are already in pixels. Give a dummy img input
        # to multiply to convert normal values -> pixels              
        img = np.zeros(shape=(1,1))

        while not self.done:
            if not self.pause:
                image_np = self.getImage().copy()
                self.class_index, self.persons, self.confidences, self.bbs = self.predict_once(image_np)
                for i, c in enumerate(self.confidences):
                    if c <= 0.5:  # 0.5 is kept as threshold for known face.
                        self.persons[i] = "_unknown"

                # Convert DLIB rectangles to format the pipeline uses.
                if len(self.bbs) > 0:
                    counter = 0
                    boxes = np.zeros(shape=(len(self.bbs),4))
                    for rect in self.bbs:
                        boxes[counter,:] = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                        counter += 1
                    classes = np.asarray(self.class_index)
                    scores = np.asarray(self.confidences) 
                    if (len(self.bbs) == 2):
                        done = True
                else:
                    boxes = np.asarray([]).reshape(-1,1)
                    scores = np.asarray([]).reshape(-1,1)
                    classes = np.asarray([])
                    self.persons = []


                self.output_data.bbs = np.asarray([])
                self.output_data.scores = np.asarray([])
                self.output_data.classes = np.asarray([])
                [boxes, scores, classes, _ ] = self.global_tracker.pipeline(boxes, scores, classes, img,
                                            0.0,FaceTracker = True)
                
                self.detect_faces(image_np)

                time.sleep(0.1)
            else:
                self.output_data.bbs = np.asarray([])
                self.output_data.scores = np.asarray([])
                self.output_data.classes = np.asarray([])
                boxes = np.asarray([]).reshape(-1,1)
                scores = np.asarray([]).reshape(-1,1)
                classes = np.asarray([])
                self.persons = []
                time.sleep(1.0)
            self.output_data.bbs = np.asarray(list(boxes))
            self.output_data.scores = np.asarray(list(scores))
            self.output_data.classes = np.asarray(list(classes))
            self.output_data.persons = list(self.persons)
            

    