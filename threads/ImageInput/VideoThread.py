import cv2
import time
from threads.ImageInput.AbstractImageInputThread \
    import AbstractImageInputThread


class VideoThread(AbstractImageInputThread):
    def __init__(
            self,
            name,
            VIDEO_SRC,
            IMAGE_WIDTH=640,
            IMAGE_HEIGHT=480,
            FPS=25.0):
        super().__init__(name,
                         IMAGE_WIDTH,
                         IMAGE_HEIGHT)
        self.cap = self.init_input(IMAGE_WIDTH, IMAGE_HEIGHT, VIDEO_SRC)
        self.SLEEP_TIME = 1.0 / FPS

    def init_input(self, IMAGE_WIDTH, IMAGE_HEIGHT, VIDEO_SRC):
        cap = cv2.VideoCapture(VIDEO_SRC)
        assert cap.isOpened(), 'Could not open video.'
        cap.set(3, IMAGE_WIDTH)
        cap.set(4, IMAGE_HEIGHT)
        return cap

    def stop(self):
        super().stop()
        self.cap.release()

    def updateImg(self, threadName):
        while not self.done:
            is_read, image_np = self.cap.read()
            if not is_read:
                self.cap.set(cv2.cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                self.image_data.image_np = image_np
            
            time.sleep(self.SLEEP_TIME)
