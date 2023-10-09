import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def get_feature(images, cluster_k):
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


'''
img = imread('./data/testdata/testSet/00002.jpg')
img = cv2.resize(img, (340, 350))

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
sift_image = cv2.drawKeypoints(imgGray, keypoints, img)

df = pd.DataFrame(descriptors)
print(df)
# plt.figure()
# plt.imshow(sift_image,cmap='gray')
# plt.savefig('./data/testdata/dog_.jpg')'''
