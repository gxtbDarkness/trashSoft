import json
import os
import random

import cv2
from imutils import paths


def load_test_images(test_set_path):
    """
    :param test_set_path: test set path
    :return: images list
    """
    images = []
    for imagePath in paths.list_images(test_set_path):
        image = cv2.imread(imagePath)
        images.append(image)

    return images


def load_images(train_set_path, classes):
    """
    :param classes: classes need to be loaded
    :param train_set_path: train set path
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


def load_image_paths(train_set_path, classes):
    """
    :param train_set_path: train set path
    :param classes: classes need to be loaded
    :return: two lists, images_path list, labels list
    """
    image_paths = []
    labels = []
    for clas in classes:
        for imagePath in paths.list_images(os.path.join(train_set_path, clas)):
            image_paths.append(imagePath)
            labels.append(clas)
            print("Loaded:" + os.path.basename(imagePath))

    return image_paths, labels


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


def show_dataset(dataset_path, classes):
    """
    :param dataset_path: dataset path
    :param classes: classes need to be counted
    :return: none
    """
    print("----------------------------------------------------------------------------------------------------------")
    print("show samples in every class.")
    sum = 0
    for clas in classes:
        sum = sum + len(os.listdir(os.path.join(dataset_path, clas)))
        print(f"Class [{clas}] contains samples:", len(os.listdir(os.path.join(dataset_path, clas))))
    print("Total samples in the dataset:", sum)
    print("----------------------------------------------------------------------------------------------------------")


def create_train_and_val(data_folder, train_ratio):
    """
    :param data_folder: dataset path
    :param train_ratio: the ratio of train_set
    :return: none
    """
    # 设置数据文件夹路径
    # data_folder = './data/RubbishClassification/trainval'
    # 初始化数据列表
    data_list = []

    # 遍历每个类别的文件夹
    for label in os.listdir(data_folder):
        label_path = os.path.join(data_folder, label)

        # 获取当前类别的所有图片文件
        image_files = [f for f in os.listdir(label_path) if f.endswith('.jpg')]

        # 打乱图片顺序
        random.shuffle(image_files)

        # 计算训练集和验证集的分割索引
        train_split_index = int(len(image_files) * train_ratio)

        # 将图片路径和标签写入数据列表
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(label_path, image_file)
            data_entry = {"dir": image_path, "label": label}

            if i < train_split_index:
                # 添加到训练集
                data_list.append(data_entry)
            else:
                # 添加到验证集
                data_list.append(data_entry)

    # 打乱数据列表
    random.shuffle(data_list)
    # 分割成训练集和验证集
    train_size = int(len(data_list) * train_ratio)
    train_data = data_list[:train_size]
    val_data = data_list[train_size:]

    # 将数据写入JSON文件
    with open('train.json', 'w') as train_file:
        json.dump(train_data, train_file)

    with open('val.json', 'w') as val_file:
        json.dump(val_data, val_file)


def create_test(folder_path):
    """
    :param folder_path:
    :return: none
    """
    # 图片文件夹路径
    # folder_path = "./data/expand_data/test_set"  # 替换为你的图片文件夹路径

    # 用于存储图片信息的列表
    image_list = []

    # 遍历文件夹中的图片文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # 只处理 jpg 格式的文件
            image_path = os.path.join(folder_path, filename)
            # 构建对应的格式
            image_info = {
                "dir": image_path,
                "label": "-1"
            }
            # 添加到图片信息列表中
            image_list.append(image_info)

    print(len(image_list))
    # 写入到 JSON 文件
    output_file = "predict.json"  # 保存的 JSON 文件路径
    with open(output_file, "w") as json_file:
        json.dump(image_list, json_file)
