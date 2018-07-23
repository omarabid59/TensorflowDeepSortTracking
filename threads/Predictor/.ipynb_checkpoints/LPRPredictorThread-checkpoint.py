from openalpr import Alpr
from SubsetImgPredictor import SubsetImgPredictor
import numpy as np
import time
from helper_data_encapsulation import OutputClassificationData
# REMOVE THIS. SHOULD NOT BE DRAWING IN THIS THREAD
import cv2
class LPRPredictorThread(SubsetImgPredictor):
    def __init__(self, name, PATH_TO_CKPT,
                 PATH_TO_LABELS,
                 SRC_DATA,
                 class_indices,
                 score_thresh,
                 WITH_TRACKER=True,
                 ENABLE_BY_DEFAULT=False):
        super().__init__(name, PATH_TO_CKPT,
                 PATH_TO_LABELS,
                 SRC_DATA,
                 class_indices,
                 score_thresh,
                 WITH_TRACKER,
                 ENABLE_BY_DEFAULT)
        # Initialize extra variables
        self.lpr_imgs = []
        self.lpr_ocr_results = []
        self.alpr = Alpr("ae", "/etc/openalpr/openalpr-no-detection.conf", "/usr/share/openalpr/runtime_data")
        
        # For displaying text
        self.fontFace                   = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10,500)
        self.fontScale              = 2.0
        self.fontColor              = (255,255,255)
        self.lineType               = 2

    def predict(self,threadName):
        while not self.done:
            if not self.pause:
                image_np = self.SRC_DATA.image_np
                [temp_bbs, temp_scores, temp_classes] = self.find_class_of_interest()
                '''
                STEP 2: For all of the bbs from above, we need to get the subset of the image to pass into the prediction function.
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
                    '''
                    Runs the OpenALPR for character recognition.
                    NOTE: THE REST OF THE CODE FOR THIS FUNCTION IS IDENTICAL TO IT"S PARENT CLASS EXCEPT FOR THIS LINE!
                    '''
                    self.runOpenALPR(image_np)
                else:
                    self.output_data.bbs = np.asarray([]) 



            else:
                self.output_data.bbs = np.asarray([])
                time.sleep(5.0) # Sleep for 2 seconds

    
    
    def runOpenALPR(self, input_np):
        '''
        Given the input image ``input_np``, extracts the subset which contains the license plate,
        runs the OpenALPR character recognition algorithm, then outputs the following.
        lpr_imgs: A list of License Plate (LP) images.
        lpr_ocr_results: A list of strings of the most likely plates.
        '''
        [h,w,_] = input_np.shape
        lpr_imgs = []
        lpr_ocr_results = []
        if len(self.output_data.bbs) > 0:
            j = 0
            for bb in self.output_data.bbs:
                # Get the Image subset
                lpr_img = input_np[int(bb[0]*h):int(bb[2]*h),
                                        int(bb[1]*w):int(bb[3]*w),:]
                # Get the image shape
                [h_lpr, w_lpr, _] = lpr_img.shape
                # Run the ALPR
                result = self.alpr.recognize_ndarray(lpr_img)
                if len(result) > 0:
                    result = result['results']
                    if len(result) > 0:
                        candidates = result[0]['candidates']
                        # For now, get the first LP detected.
                        plate = candidates[0]['plate']
                        lpr_ocr_results.append(plate)
                lpr_imgs.append(lpr_img)
        self.lpr_imgs = lpr_imgs
        self.lpr_ocr_results = lpr_ocr_results

    def generateOCRImages(self):
        '''
        Takes as input a string array of OCR results and outputs it as an image, ready to be displayed onto the screen.
        OUTPUT:
            lpr_imgs_ocr: Output LPR image.
        '''
        #print('THIS SHOULD ONLY RETURN THE LPR IMG TEXT')
        lpr_imgs_ocr_output = []
        for idx, img in enumerate(self.lpr_imgs):
            lpr_img_ocr = np.zeros(shape=(50,200,3))

            if idx < len(self.lpr_ocr_results):
                cv2.putText(lpr_img_ocr,self.lpr_ocr_results[idx],(10,30),
                            self.fontFace,
                            self.fontScale,
                            self.fontColor,
                            self.lineType)
            else:
                cv2.putText(lpr_img_ocr,'------',(10,30),
                            self.fontFace,
                            self.fontScale,
                            self.fontColor,
                            self.lineType)
            lpr_imgs_ocr_output.append(lpr_img_ocr)
        ocr_strings = list(self.lpr_ocr_results)
        lpr_imgs = list(self.lpr_imgs)
        return [lpr_imgs, lpr_imgs_ocr_output, ocr_strings]
    
    def generateSingleOCRImage(self,text):
        '''
        Generates a OCR text image array with a single image.
        '''
        lpr_img_ocr = np.zeros(shape=(60,260,3))
        cv2.putText(lpr_img_ocr,text,(10,50),
                            self.fontFace,
                            self.fontScale,
                            self.fontColor,
                            self.lineType)
       
        return lpr_img_ocr
    
    