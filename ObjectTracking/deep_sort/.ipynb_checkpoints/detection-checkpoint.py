# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlbr : ndarray
        Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlbr, score, image_np):
        feature = [1]
        self.height = image_np.shape[0]
        self.width = image_np.shape[1]
        
        self.tlbr = [tlbr[1]*self.width, tlbr[0]*self.height,
                     tlbr[3]*self.width, tlbr[2]*self.height]
        self.confidence = float(score)
        self.feature = np.asarray(feature, dtype=np.float32)
        
        # TLWH Conversion
        tlwh = [self.tlbr[0], self.tlbr[1],
               # Width
               self.tlbr[2] - self.tlbr[0],
                # Height
                self.tlbr[3] - self.tlbr[1]]
        #tlwh = [tlbr[0], tlbr[1],
        #       tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]]
        self.tlwh = np.asarray(tlwh, dtype=np.float)


    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        return self.tlbr

    def to_tlwh(self):
        '''
        Bounding box in format `(top left x, top left y, width, height)`.
        '''
        return self.tlwh
    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
