import os
import cv2
from imutils import paths


def load_images(train_set_path, classes):
    """
    :param classes: classes need to loaded
    :param train_set_path:
    :return: a list, containing key value pairs (image, label)
    """
    images = []
    labels = []
    for clas in classes:
        for imagePath in paths.list_images(os.path.join(train_set_path, clas)):
            image = cv2.imread(imagePath)
            images.append(image)
            labels.append(clas)
            print("Loaded:" + os.path.basename(imagePath))

    return join_data(images, labels)


def separate_data(images):
    """
    :param images: a list tuple (image, label)
    :return: list of data and list of labels
    """
    return zip(*images)


def join_data(images, labels):
    """
    :param images
    :param labels
    :return: list of tuples
    """
    return zip(images, labels)

