import cv2
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
import utils
import random
import numpy as np
from sklearn.cluster import KMeans


# 提取STFT特征矩阵与关键点数量
def sift_features(images):
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
    image_paths = [item[0] for item in datas]
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        images.append(img)
    descriptors, keys = sift_features(images)
    descs = descriptors[0]
    for desc in descriptors[1:]:
        if desc is not None:
            descs = np.vstack((descs, desc))
    kmeans.fit(descs.astype(float))
    return kmeans


# 利用获取的K-Means模型输出矩阵的聚类矩阵
def get_kmeans_feature(kmeans, image_paths):
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        images.append(img)
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


def main():
    # 加载数据
    # classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    classes = ["0", "1"]

    root_dir = "./data/RubbishClassification/trainval"
    images_labels = list(utils.separate_data(utils.load_image_paths(root_dir, classes)))
    image_paths, labels = [item[0] for item in images_labels], [item[1] for item in images_labels]

    # 创建K-Means模型
    kmeans = KMeans(n_clusters=200, random_state=42, n_init=10)

    # 将数据分批次处理,batch_size为一个batch中包含的元素个数
    batch_size = 64
    random.shuffle(images_labels)
    batchs = [images_labels[i:i + batch_size] for i in range(0, len(images_labels), batch_size)]

    for batch_idx, batch in enumerate(batchs):
        # print(batch)
        kmeans = train_kmeans(kmeans, batch)
        print("K-Means: Trained batch", batch_idx)

    # 保存K-Means模型
    joblib.dump(kmeans, 'kmeans_model.pkl')

    kmeans = joblib.load('kmeans_model.pkl')
    print(image_paths)

    # 用训练出的K-Means获得聚类矩阵
    feature = get_kmeans_feature(kmeans, image_paths)
    # 进行训练
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(feature, labels)

    # 进行验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # 执行 k 折交叉验证并计算准确率
    scores = cross_val_score(model, feature, labels, cv=kf, scoring='accuracy')
    # 输出每次交叉验证的准确率和平均准确率
    for i, score in enumerate(scores):
        print(f'Fold {i + 1}: {score:.4f}')
    print(f'Mean Accuracy: {scores.mean():.4f}')


if __name__ == "__main__":
    main()
