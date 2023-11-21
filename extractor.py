import random

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from skimage import feature
from sklearn.cluster import KMeans

EPS = 1e-7


def get_hog_feature(image):
    """
    Helper function to get hog feature of image
    :param image: image
    :return: hog feature of image
    """
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (128, 128))
    hist = hog(gray, cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
    return hist


def get_lbp_feature(image, numpoints, radius):
    """
    Helper function to get circle lbp feature of image
    :param radius: circle radius
    :param numpoints: num of sample points
    :param image: image
    :return: lbp feature of image
    """
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


def sift_features(images):
    """
    :param images: images
    :return: sift descriptors of images
    """
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )
    descriptors = []
    keys = []
    for img in images:
        key, desc = sift.detectAndCompute(img, None)
        if desc is None:
            # 如果没有特征矩阵，将其置为0矩阵
            desc = np.zeros((1, 128), dtype=np.float32)
            print("None desc!!!!!!")
        descriptors.append(desc)
        keys.append(key)
    return descriptors, keys


# batch中的矩阵进行堆栈，进行K-Means聚类，获取K-Means模型
def train_kmeans(kmeans, datas):
    """
    :param kmeans: kmeans model
    :param datas: training data
    :return: kmeans model
    """
    images = [item[0] for item in datas]
    descriptors, keys = sift_features(images)
    descs = descriptors[0]
    for desc in descriptors[1:]:
        if desc is not None:
            descs = np.vstack((descs, desc))
    kmeans.fit(descs.astype(float))
    return kmeans


# 利用获取的K-Means模型输出矩阵的聚类矩阵
def get_kmeans_feature(kmeans, images):
    """
    :param kmeans: kmeans model
    :param images: images to be predicted
    :return: sift features of images
    """
    descriptors, keys = sift_features(images)
    histo_all = []
    for key, desc in zip(keys, descriptors):
        histo = np.zeros(200)
        nkp = np.size(len(key))
        for d in desc:
            idx = kmeans.predict([d])
            # instead of increasing each bin by one, add the normalized value
            histo[idx] += 1 / nkp
        histo_all.append(histo)
    return histo_all


def get_sift_feature(images, labels, cluster_k):
    """
    Helper function to get sift features of images
    :param labels: labels of images
    :param cluster_k: Clustering coefficient
    :param images: images
    :return: sift features of images
    """
    images_labels = list(zip(images, labels))

    # 创建K-Means模型
    kmeans = KMeans(n_clusters=cluster_k, random_state=42, n_init=10)

    # 将数据分批次处理,batch_size为一个batch中包含的元素个数
    batch_size = 64
    random.shuffle(images_labels)
    batchs = [images_labels[i:i + batch_size] for i in range(0, len(images_labels), batch_size)]

    for batch_idx, batch in enumerate(batchs):
        kmeans = train_kmeans(kmeans, batch)
        print("K-Means: Trained batch", batch_idx)

    # 保存K-Means模型
    joblib.dump(kmeans, 'kmeans_model.pkl')
    kmeans = joblib.load('kmeans_model.pkl')

    # 用训练出的K-Means获得聚类矩阵
    hists = get_kmeans_feature(kmeans, images)
    return hists

