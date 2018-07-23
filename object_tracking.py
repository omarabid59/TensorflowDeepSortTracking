import cv2
import warnings
warnings.filterwarnings('ignore')
from ObjectTracking.deep_sort_tracker import DeepSortTracker
from utilities import constants
from utilities import helper
from threads.ImageInput.WebcamThread import WebcamThread
from threads.Predictor.PredictorImage import PredictorImage


print('Running a Tensorflow model with the DeepSORT Tracker')

# Run the webcam thread
thread_image = WebcamThread('Webcam Thread',1)
thread_image.start()
image_data = thread_image.image_data
# Run the COCO Model
thread_coco = PredictorImage('coco',
                       constants.CKPT_COCO,
                       constants.LABELS_COCO,
                       image_data,
                       score_thresh = 0.5,
                       WITH_TRACKER = False)
thread_coco.start()
thread_coco.continue_predictor()
# Initialize the Tracker
tracker = DeepSortTracker()

# Run the main loop
while True:
    # Grab the image and convert from RGB -> BGR
    image_np = thread_image.image_data.image_np.copy()[:,:,::-1]
    output_data = thread_coco.output_data
    output_data = tracker.run(output_data,image_np)
    image_np = helper.drawDetectedBBs(image_np.copy(),
                                 output_data,
                               score_thresh = 0.1)

    
    
    frameName = 'Main Frame'
    cv2.imshow(frameName, image_np)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
cv2.destroyAllWindows()