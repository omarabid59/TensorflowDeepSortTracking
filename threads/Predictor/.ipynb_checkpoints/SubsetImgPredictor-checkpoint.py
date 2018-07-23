from AbstractSubsetImgPredictor import AbstractSubsetImgPredictor
import numpy as np
import time
class SubsetImgPredictor(AbstractSubsetImgPredictor):
    def __init__(self, name, PATH_TO_CKPT,
                 PATH_TO_LABELS,
                 SRC_DATA,
                 class_indices,
                 score_thresh,
                 IMG_SCALE = 1.0,
                 WITH_TRACKER=True,
                 ENABLE_BY_DEFAULT=False):
        super().__init__(name, PATH_TO_CKPT,
                 PATH_TO_LABELS,
                 SRC_DATA,
                 class_indices,
                 score_thresh,
                 IMG_SCALE,
                 WITH_TRACKER,
                 ENABLE_BY_DEFAULT)
        [self.detection_graph, self.sess] = self.load_model(PATH_TO_CKPT)

        [self.image_tensor, self.boxes_tensor, self.scores_tensor, self.classes_tensor, self.num_detections_tensor] = self.getTensors()  
        self.output_data.category_index = self.category_index
        
    def getTensors(self):
        '''
        Returns the tensors 
        '''
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        return [image_tensor, boxes, scores, classes, num_detections]   
  
    def predict(self,threadName):
        while not self.done:
            if not self.pause:
                image_np = self.getImage()
                # Step 1. Find all the classes of interest and their bounding boxes.
                [temp_bbs, temp_scores, temp_classes] = self.find_class_of_interest()
                '''
                STEP 2: For all of the bbs from above, we need to get the subset of the image to pass into the  prediction function.
                Then, run the prediction function on each of these images. Append these into the output frame
                '''
                bbs = []
                scores = []
                classes = []               
                for bb in temp_bbs:
                    # Get the bounding box coordinates
                    [fixed_bb, cropped_img] = self.getImgSubset(bb,image_np)
                    


                    # Run the prediction function on each of these images.
                    [temp_box, temp_score, temp_class] = self.predict_once(cropped_img.copy())
                    # Next, we need to remap the bb coordinates
                    [temp_box, temp_score, temp_class] = self.remap_bb_coords(cropped_img, image_np, temp_box, temp_score,                                                                                          temp_class, fixed_bb)
                    if len(temp_box) > 0:
                        bbs.append(temp_box)
                        scores.append(temp_score)
                        classes.append(temp_class) 


                if len(bbs) > 0:
                    # Since we end up with a list of arrays, concatenate them together.
                    bbs = np.concatenate(bbs)
                    assert(bbs.ndim == 2), self.name + '. predict(). Error. The dimensionality should be 2.' + str(bbs)
                    scores = np.concatenate(scores)
                    classes = np.concatenate(classes)
                    # A bit of a hack. If we have 3 dimensions, squeeze it.
                    if bbs.ndim == 3:
                        print("DIMENSIONS: " + str(self.output_data.bbs.ndim))
                        self.output_data.bbs = np.squeeze(bbs)
                        self.output_data.scores = np.squeeze(scores)
                        self.output_data.classes = np.squeeze(classes)  
                        if self.output_data.bbs.ndim == 3:
                            print("DIMENSIONS: " + str(self.output_data.bbs.ndim))
                    else:
                        self.output_data.bbs = bbs
                        self.output_data.scores = scores
                        self.output_data.classes = classes                        
                    
             
                    #print('Subset Predictor BBS: ' + str(self.output_frame.bbs))
                else:
                    self.output_data.bbs = np.asarray([]) 

                        
                        
            else:
                self.output_data.bbs = np.asarray([])
                time.sleep(2.0) # Sleep for 2 seconds
    def predict_once(self,image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        with self.sess.as_default(), self.detection_graph.as_default():
            [temp_box, temp_score, temp_class,temp_num_detections] = self.sess.run(
                                                                                [self.boxes_tensor, self.scores_tensor, self.classes_tensor, self.num_detections_tensor],
                                                                                 feed_dict={self.image_tensor: image_np_expanded})
            if self.WITH_TRACKER:
                [temp_box, temp_score, temp_class,
                    _] = self.global_tracker.pipeline(temp_box,
                                                      temp_score,
                                                      temp_class,
                                                      image_np,
                                                      score_thresh=self.score_thresh)
            else:
                temp_box = np.squeeze(temp_box)
                temp_score = np.squeeze(temp_score)
                temp_class = np.squeeze(temp_class)
                
            if temp_box == ():
                temp_box = np.asarray([])
            time.sleep(0.1)
            return [temp_box, temp_score, temp_class]
  
