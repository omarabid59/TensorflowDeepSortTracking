from ObjectTracking.deep_sort_tracker import DeepSortTracker
from threads.ImageInput.WebcamThread import WebcamThread
from threads.ImageInput.VideoThread import VideoThread
from threads.Predictor.PredictorImage import PredictorImage
import cv2
import warnings
from utilities import constants
from utilities import helper
import argparse

warnings.filterwarnings('ignore')
WEBCAM_INPUT = 'cam'


def init(inputSrc):
    if inputSrc == WEBCAM_INPUT:
        # Run the webcam thread
        thread_image = WebcamThread('Webcam Thread', 1)
    else:
        thread_image = VideoThread('Video Thread', inputSrc, FPS=25.0)

    thread_image.start()
    image_data = thread_image.image_data
    # Run the COCO Model
    thread_coco = PredictorImage('coco',
                                 constants.CKPT_COCO,
                                 constants.LABELS_COCO,
                                 image_data,
                                 score_thresh=0.5,
                                 WITH_TRACKER=False)
    thread_coco.start()
    thread_coco.continue_predictor()
    # Initialize the Tracker
    tracker = DeepSortTracker()
    return tracker, thread_coco, thread_image


def main(tracker, thread_coco, thread_image):
    frameName = 'Main Frame'
    print('Running a Tensorflow model with the DeepSORT Tracker')
    # Run the main loop
    while True:
        # Grab the image and convert from RGB -> BGR
        image_np = thread_image.image_data.image_np.copy()[:, :, ::-1]
        output_data = thread_coco.output_data
        output_data = tracker.run(output_data, image_np)
        image_np = helper.drawDetectedBBs(image_np.copy(),
                                          output_data,
                                          score_thresh=0.1)

        cv2.imshow(frameName, image_np)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SSD object detection with DeepSORT tracking')
    parser.add_argument('--input', default='cam',
                        help='"cam" for Webcam or video file path')

    args = parser.parse_args()

    tracker, thread_coco, thread_image = init(args.input)
    main(tracker, thread_coco, thread_image)
