#!/usr/bin/env python3
# import the necessary packages
import sys
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

import extractor
import utils
from sklearn.svm import SVC
import argparse


def strbool(v):
    """
    Helper function to parse different expressions to True or False
    :param v: value
    :return: True or False, depending on the input
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def lbp(images):
    """
    Helper function to get lbp features of images
    :param images: image list
    :return: lbp features of images
    """
    data = []
    for image in images:
        data.append(extractor.get_lbp_feature(image, 24, 8))
    return data


def hog(images):
    """
    Helper function to get hog features of images
    :param images: image list
    :return: hog features of images
    """
    data = []
    for image in images:
        data.append(extractor.get_hog_feature(image))
    return data


def sift(images, labels, k):
    """
    Helper function to get sift features of images
    :param labels: labels of images
    :param k: Clustering coefficient
    :param images: image list
    :return: sift features of images
    """
    data = extractor.get_sift_feature(images, labels, k)
    return data


def get_SVM():
    """
    Helper function to get SVM model
    :return: SVM model
    """
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    return model


def get_LogisticRegression():
    """
    Helper function to get LogisticRegression model
    :return: LogisticRegression model
    """
    model = LogisticRegression(multi_class='ovr', max_iter=10000, solver='lbfgs',
                               C=1.0, class_weight=None, random_state=42, n_jobs=-1)
    return model


def get_MLPClassifier():
    """
    Helper function to get MLPClassifier
    :return: MLPClassifier
    """
    model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                          solver='adam', max_iter=1000, random_state=42)
    return model


def main():
    # 命令行设置
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="Method to use. Available: SIFT, HOG, LBP")
    parser.add_argument("-classifier", help="Classifier to use. Available: SVM, LOG, MLP")
    parser.add_argument('-c', '--classes', nargs='+', help='<Required> Which classes to load', required=True)
    parser.add_argument('-k', '--k', type=int, default=200, help='Define number of clusters')
    parser.add_argument('-s', '--splits', type=int, default=5, help='Define number of KFold splits')
    parser.add_argument('-cval', '--crossval', type=strbool, nargs='?', const=True, default=True,
                        help='Set True for using cross validation')

    args = parser.parse_args()

    if not args.method:
        parser.print_help()
        sys.exit()

    # 加载图片
    images, labels = utils.separate_data(utils.load_images("./data/RubbishClassification/trainval", args.classes))

    # 判断使用的方法类型,分别进行特征提取，返回特征向量
    feature = []
    if args.method.lower() == "lbp":
        feature = lbp(images)
    elif args.method.lower() == "hog":
        feature = hog(images)
    elif args.method.lower() == "sift":
        feature = sift(images, labels, args.k)
    else:
        parser.print_help()
        sys.exit()

    # 判断使用的模型类型,返回对应模型
    if args.classifier.lower() == "svm":
        model = get_SVM()
    elif args.classifier.lower() == "log":
        model = get_LogisticRegression()
    elif args.classifier.lower() == "mlp":
        model = get_MLPClassifier()
    else:
        parser.print_help()
        sys.exit()

    # 进行训练
    model.fit(feature, labels)

    if args.crossval:
        # 进行验证
        kf = KFold(n_splits=args.splits, shuffle=True, random_state=42)
        # 执行 k 折交叉验证并计算准确率
        scores = cross_val_score(model, feature, labels, cv=kf, scoring='accuracy')
        # 输出每次交叉验证的准确率和平均准确率
        for i, score in enumerate(scores):
            print(f'Fold {i + 1}: {score:.4f}')
        print(f'Mean Accuracy: {scores.mean():.4f}')
    else:
        # 加载图片
        images = utils.separate_data(utils.load_test_images("./data/RubbishClassification/test_set"))
        predictions = model.predict(images)
        for i in range(len(images)):
            print(f"第{i}个图像的类别是：", predictions[i])


if __name__ == "__main__":
    main()
