import cv2
from threads.ImageInput.AbstractImageInputThread import AbstractImageInputThread
class WebcamThread(AbstractImageInputThread):
    def __init__(self, name, VIDEO_ID,IMAGE_WIDTH = 640 ,IMAGE_HEIGHT = 480):
        super().__init__(name,
                        IMAGE_WIDTH,
                        IMAGE_HEIGHT)
        self.cap = self.init_input(IMAGE_WIDTH,IMAGE_HEIGHT, VIDEO_ID)
    
    def init_input(self, IMAGE_WIDTH,IMAGE_HEIGHT, VIDEO_ID):
        cap = cv2.VideoCapture(VIDEO_ID)
        assert cap.isOpened() == True, 'Could not open Webcam.'
        cap.set(3, IMAGE_WIDTH)
        cap.set(4, IMAGE_HEIGHT)
        return cap
    def stop(self):
        super().stop()
        self.cap.release()

