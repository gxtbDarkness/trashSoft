import argparse
import os.path
import sys

from torchvision import models
import torch
import json
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader


def strbool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def default_loader(path):
    return Image.open(path).resize((224, 224)).convert('RGB')


# 准备数据
class TrashSortDataset(Dataset):
    def __init__(self, data_dir, transform=None, loader=default_loader):
        super(TrashSortDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.loader = loader
        # 打开文件以读取文本内容
        with open(data_dir, 'r') as file:
            content = file.read()
        self.data_files = json.loads(content)

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data_files)

    def __getitem__(self, index):
        # Load and return a sample at the given index
        if os.path.exists(self.data_files[index]["dir"]):
            image_path = self.data_files[index]["dir"]
        else:
            image_path = os.path.join(".\\data\\RubbishClassification\\", self.data_files[index]["dir"])
        label = self.data_files[index]["label"]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
            image = image.float()
        return image, torch.tensor(int(label))


def saveModel(model):
    path = "./result/deeplearning/trashSortModel.pth"
    torch.save(model.state_dict(), path)


def valAccuracy(model, device, val_loader, loss_fn, val_loss):
    model.eval()
    accuracy = 0.0
    running_loss = 0.0
    total = 0.0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all val images
    val_loss.append(running_loss / 1000.0)
    accuracy = (100 * accuracy / total)
    return accuracy


def calculate_label_accuracies(model, device, val_loader, num_classes):
    model.eval()
    label_correct = {i: 0 for i in range(num_classes)}
    label_total = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(num_classes):
                label_total[i] += (labels == i).sum().item()
                label_correct[i] += ((predicted == labels) & (labels == i)).sum().item()

    label_accuracies = {i: label_correct[i] / label_total[i] if label_total[i] > 0 else 0.0 for i in range(num_classes)}
    with open('./result/deeplearning/classes_accuracy.json', 'a') as f:
        json.dump(label_accuracies, f)
        f.write('\n')
    return label_accuracies


def train(model, device, optimizer, loss_fn, train_loader, val_loader, num_epochs, classes_num):
    best_accuracy = 0.0
    train_loss = []
    val_loss = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            # 获取输入
            images = images.to(device)
            labels = labels.to(device)

            # 梯度归0
            optimizer.zero_grad()
            # 预测训练集中图像所属的类
            outputs = model(images)
            # 基于输出与标签计算损失
            loss = loss_fn(outputs, labels)
            # 反向传播
            loss.backward()
            # 基于计算出的梯度调整参数
            optimizer.step()

            running_loss += loss.item()  # extract the loss value

        train_loss.append(running_loss / 1000.0)
        calculate_label_accuracies(model, device, val_loader, classes_num)
        accuracy = valAccuracy(model, device, val_loader, loss_fn, val_loss)
        print('For epoch', epoch + 1, 'the val accuracy over the whole val set is %d %%' % (accuracy))

        # 保存准确率最高的模型
        if accuracy > best_accuracy:
            saveModel(model)
            best_accuracy = accuracy

    with open('./result/deeplearning/loss.json', 'a') as f:
        json.dump({"train_loss": train_loss}, f)
        f.write('\n')
        json.dump({"val_loss": val_loss}, f)
        f.write('\n\n')
    return train_loss, val_loss


def show_loss(train_loss, val_loss, epoch):
    epochs = list(range(1, epoch + 1))
    # 绘制训练集损失曲线
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    # 绘制验证集损失曲线
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    # 添加标签和标题
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    # 添加图例
    plt.legend()
    plt.savefig('loss_chart.png', format='png')
    # 显示图形
    plt.show()


def clear():
    with open('./result/deeplearning/classes_accuracy.json', 'w') as f:
        json.dump({}, f)

    with open('./result/deeplearning/loss.json', 'w') as f:
        json.dump({}, f)


def main():
    # 命令行设置
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model to use. Available: AlexNet, ResNet")
    parser.add_argument('-pretrained', '--pre', type=strbool, nargs='?', const=True, default=True,
                        help='Set True for Pretrain')
    parser.add_argument('-extensive_dataset', '--d', type=strbool, default=False, help='Decide if choose the extensive '
                                                                                       'dataset')
    parser.add_argument('-batch_size', '--b', type=int, help='batch_size eg. 64, 32, 16, 8', default=32)
    parser.add_argument('-learning_rate', '--lr', type=float, help='set the learning rate', default=0.0001)
    parser.add_argument('-epochs', '--ep', type=int, help='set the epochs', default=100)
    parser.add_argument('-classes_num', '--cla', type=int, help='set number of classes', default=16)
    args = parser.parse_args()

    # 检查是否选择模型
    if not args.model:
        parser.print_help()
        sys.exit()

    # 选择数据集
    if not args.d:
        train_dir = "./data/RubbishClassification/train_.json"
        val_dir = "./data/RubbishClassification/val_.json"
    else:
        train_dir = "./data/expand_data/train_.json"
        val_dir = "./data/expand_data/val_.json"

    # 加载数据集与测试集
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = TrashSortDataset(train_dir, transform=transformations)
    train_loader = DataLoader(train_set, batch_size=args.b, shuffle=True, num_workers=0)
    val_set = TrashSortDataset(val_dir, transform=transformations)
    val_loader = DataLoader(val_set, batch_size=args.b, shuffle=False, num_workers=0)

    # 打印数据集的长度（总样本数）
    print("Total samples in the dataset:", len(train_set))

    # 初始化神经网络模型
    if args.model.lower() == 'alexnet':
        model = models.alexnet(pretrained=args.pre)
    elif args.model.lower() == 'resnet':
        model = models.resnet50(pretrained=args.pre)
        fc_inputs = model.fc.in_features
        model.fc = nn.Linear(fc_inputs, args.cla)
    else:
        parser.print_help()
        sys.exit()

    # 清空json文件中的内容
    clear()

    # 使用GPU运行
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("The model will be running on", device, "device")

    # 定义损失函数与优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    # 训练模型并获取训练集与测试集损失
    train_loss, val_loss = train(model, device, optimizer, loss_fn, train_loader, val_loader, args.ep, args.cla)

    # 绘图
    show_loss(train_loss, val_loss, args.ep)


if __name__ == "__main__":
    main()
