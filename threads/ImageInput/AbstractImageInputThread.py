import threading
from abc import ABC, abstractmethod


class ImageData:
    def __init__(self):
        self.image_np = ()


class AbstractImageInputThread(threading.Thread, ABC):

    @abstractmethod
    def __init__(self, name, IMAGE_WIDTH=640, IMAGE_HEIGHT=480):
        threading.Thread.__init__(self)
        ABC.__init__(self)
        self.name = name
        self.image_data = ImageData()
        self.done = False
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.cap = []

    def run(self):
        print("Starting " + self.name)
        self.updateImg(self.name)
        print("Exiting " + self.name)

    def updateImg(self, threadName):
        while not self.done:
            _, self.image_data.image_np = self.cap.read()

    def getImage(self):
        return self.image_data.image_np

    def stop(self):
        self.done = True
