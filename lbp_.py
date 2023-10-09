import cv2
from skimage import feature
import numpy as np

EPS = 1e-7


def get_feature(image, numpoints, radius):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, numpoints,
                                       radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numpoints + 3),
                             range=(0, numpoints + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + EPS)
    return hist
