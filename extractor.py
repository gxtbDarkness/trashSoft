import cv2
import numpy as np
from skimage.feature import hog
from skimage import feature
from sklearn.cluster import MiniBatchKMeans

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


def get_sift_feature(images, cluster_k):
    """
    Helper function to get sift features of images
    :param cluster_k:
    :param images: images
    :return: sift features of images
    """
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )
    cluster = MiniBatchKMeans(n_clusters=cluster_k, init_size=3 * cluster_k, random_state=0, batch_size=6)

    print("Identifying descriptors..")
    descriptors = []
    keys = []
    for img in images:
        key, desc = sift.detectAndCompute(img, None)
        descriptors.append(desc)
        keys.append(key)

    descs = descriptors[0]
    for desc in descriptors[1:]:
        if desc is not None:
            descs = np.vstack((descs, desc))
    print("Clustering data..")
    cluster.fit(descs.astype(float))
    print("Clustered " + str(len(cluster.cluster_centers_)) + " centroids")

    print("Calculating histograms for " + str(len(images)) + " items.")
    histo_all = []
    for key, desc in zip(keys, descriptors):
        histo = np.zeros(cluster_k)
        nkp = np.size(len(key))

        for d in desc:
            idx = cluster.predict([d])
            # instead of increasing each bin by one, add the normalized value
            histo[idx] += 1 / nkp
        histo_all.append(histo)
    return histo_all
