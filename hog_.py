import cv2
from skimage.feature import hog


def get_feature(image):
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 128))
    hist = hog(gray, cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
    return hist
