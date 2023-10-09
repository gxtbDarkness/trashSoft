#!/usr/bin/env python3
# import the necessary packages
import sys
from skimage.feature import hog
from sklearn.model_selection import KFold, cross_val_score

import extractor
import utils
from sklearn.svm import LinearSVC, SVC
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
        # 得到 image 的特征
        data.append(extractor.get_hog_feature(image))
    return data


def sift(images, k):
    """
    Helper function to get sift features of images
    :param images: image list
    :return: sift features of images
    """
    data = extractor.get_sift_feature(images, k)
    return data


def main():
    # 命令行设置
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="Method to use. Available: SIFT, HOG, LBP, CNN")
    parser.add_argument('-c', '--classes', nargs='+', help='<Required> Which classes to load', required=True)
    parser.add_argument('-k', '--k', type=int, default=100, help='Define number of clusters')
    parser.add_argument('-s', '--splits', type=int, default=3, help='Define number of KFold splits')
    '''parser.add_argument('-cval', '--crossval', type=strbool, nargs='?', const=True, default=True,
                       help='Set True for using cross validation')'''

    args = parser.parse_args()

    if not args.method:
        parser.print_help()
        sys.exit()

    # 加载图片
    images, labels = utils.separate_data(utils.load_images("./data/trainval", args.classes))

    # 判断使用的方法类型,分别进行特征提取，返回特征向量
    feature = []
    if args.method.lower() == "lbp":
        feature = lbp(images)
    elif args.method.lower() == "hog":
        feature = hog(images)
    elif args.method.lower() == "sift":
        feature = sift(images, args.k)
    else:
        parser.print_help()
        sys.exit()

    # 进行训练
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    # model.fit(feature, labels)

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
