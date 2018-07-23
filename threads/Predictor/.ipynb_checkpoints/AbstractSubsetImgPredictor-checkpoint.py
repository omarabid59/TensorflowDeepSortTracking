from helper_data_encapsulation import OutputClassificationData
from AbstractPredictor import AbstractPredictor
import time
import numpy as np
class AbstractSubsetImgPredictor(AbstractPredictor):
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
                         IMG_SCALE,
                         WITH_TRACKER,
                         ENABLE_BY_DEFAULT)
        
        self.output_data = OutputClassificationData()
        
        # Class Index we wish to perform the detection on.
        self.class_indices = class_indices
        
        self.SRC_DATA = SRC_DATA
        self.score_thresh = score_thresh
        
       

        
        
    def getImage(self):
        if self.IMG_SCALE < 1.0:
            self.output_data.image_np =  cv2.resize(self.SRC_DATA.image_np.copy(),
                                                (0,0), fx=self.IMG_SCALE, fy=self.IMG_SCALE) 
        else:
            self.output_data.image_np = self.SRC_DATA.image_np

        return self.output_data.image_np
    
    def getImgSubset(self, bb, image_np):
        '''
        Get the subset of the image with the given bounding boxes. We take a slightly
        larger area provided by the offset.
        '''
        fixed_bb = bb.copy()
        offset_b = .95
        offset_e = 1.05
        fixed_bb[0] = max(fixed_bb[0]*offset_b, 0.0)
        fixed_bb[1] = max(fixed_bb[1]*offset_b, 0.0)
        fixed_bb[2] = min(fixed_bb[2]*offset_e, 1.0)
        fixed_bb[3] = min(fixed_bb[3]*offset_e, 1.0)
        # Extract a subset of the image.
        [h,w,_] = image_np.shape
        cropped_img = image_np[int(fixed_bb[0]*h):int(fixed_bb[2]*h),
            int(fixed_bb[1]*w):int(fixed_bb[3]*w)]
        return [fixed_bb, cropped_img]
    def remap_bb_coords(self, cropped_img, full_size_image, bbs, scores, classes, fixed_bb):
        '''
        Remaps the bounding box coordinates of the subset image so we can plot it on the entire image once again.
        fixed_bb: The coordinates of the cropped image of where it was extracted.
        '''
        remapped_bbs = bbs.copy()
        output_bbs = []
        output_scores = []
        output_classes = []
        # Recompute the bounding box coordinates
        for remapped_bb, score, class_ in zip(bbs, scores, classes):
            [h,w,_] = full_size_image.shape
            if remapped_bb[0] > 0.0 and score > self.score_thresh:
                remapped_bb[1] = min(fixed_bb[1] + (remapped_bb[1]*cropped_img.shape[1])/full_size_image.shape[1], 1.0)
                remapped_bb[3] = min(fixed_bb[1] + (remapped_bb[3]*cropped_img.shape[1])/full_size_image.shape[1], 1.0)

                remapped_bb[0] = min(fixed_bb[0] + (remapped_bb[0]*cropped_img.shape[0])/full_size_image.shape[0], 1.0)
                remapped_bb[2] = min(fixed_bb[0] + (remapped_bb[2]*cropped_img.shape[0])/full_size_image.shape[0], 1.0)
                output_bbs.append(remapped_bb)
                output_scores.append(score)
                output_classes.append(class_)
            #else:
            #    print('None Found!')
        if len(output_bbs) > 0:        
            output_bbs = np.array(output_bbs)
            assert(output_bbs.ndim == 2), self.name + '. remap_bb_coords(). Error. The dimensionality should be 2.' + str(output_bbs)
            output_scores = np.concatenate([output_scores])
            output_classes = np.concatenate([output_classes])

        else:
            output_bbs = np.asarray([]) 
        return [output_bbs, output_scores, output_classes]
    
    def find_class_of_interest (self):
        '''
        STEP 1: Check if the class of interest has been detected. If so, find all the bounding boxes for this class.
        '''
        temp_bbs = []
        temp_scores = []
        temp_classes = []
        # Find all the bounding boxes of the person index in the class
        for bb, score, class_ in zip(self.SRC_DATA.bbs, 
                                     self.SRC_DATA.scores, 
                                     self.SRC_DATA.classes):
            if bb[0] > 0.0 and score > self.SRC_DATA.score_thresh and class_ in self.class_indices:
                temp_bbs.append(bb)
                temp_scores.append(score)
                temp_classes.append(class_)
        time.sleep(0.05)
        return [temp_bbs, temp_scores, temp_classes]
    
   