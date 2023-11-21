import argparse
import os
import sys
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


def main():
    # 命令行设置
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model to use. Available: AlexNet, ResNet")
    parser.add_argument("-model_path", "--path", help="the filepath of the model to be loaded", required=True)
    args = parser.parse_args()

    # 初始化神经网络模型
    if args.model.lower() == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif args.model.lower() == 'resnet':
        model = models.resnet50(pretrained=True)
        fc_inputs = model.fc.in_features
        model.fc = nn.Linear(fc_inputs, 16)
    else:
        parser.print_help()
        sys.exit()

    for param in model.parameters():
        param.requires_grad = False

    model.load_state_dict(torch.load(args.path), strict=False)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 文件夹路径
    folder_path = './data/expand_data/test_set'

    # 获取文件夹中所有图片文件
    image_files = [f for f in os.listdir(folder_path)]

    # 打开文件以写入预测结果
    with open('./result/deeplearning/output_file.txt', 'w') as file:
        # 遍历每张图片并进行预测
        for image_file in image_files:
            # 拼接图片路径
            image_path = os.path.join(folder_path, image_file)

            # 读取待预测的图像
            input_image = Image.open(image_path)

            # 预处理图像
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)  # 添加一个维度来创建批次（batch）

            # 如果可用GPU，将图像移至GPU上
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            input_batch = input_batch.to(device)
            model.to(device)

            # 获取预测结果
            with torch.no_grad():
                output = model(input_batch)

            # 获取预测的类别
            _, predicted = torch.max(output, 1)
            predicted_label = predicted.item()  # 将tensor转换为Python中的标量值

            # 将预测结果逐行写入到txt文件中
            file.write(f"{image_file} {predicted_label}\n")


if __name__ == "__main__":
    main()
