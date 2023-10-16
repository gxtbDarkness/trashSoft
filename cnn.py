import os
import torch
import json
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

"""import cv2
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt"""


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
        image_path = os.path.join(".\\data\\RubbishClassification", self.data_files[index]["dir"])
        label = self.data_files[index]["label"]
        # label = torch.LongTensor([int(self.data_files[index]["label"])])
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
            image = image.float()
        return image, torch.tensor(int(label))


# 搭建神经网络结构
class CNN(nn.Module):
    def __init__(self, num_classes: int = 16, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def saveModel(model):
    path = "./trashSortModel.pth"
    torch.save(model.state_dict(), path)


def testAccuracy(model, test_loader):
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return (accuracy)


def train(model, optimizer, loss_fn, train_loader, test_loader, num_epochs):
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):

            # get the inputs
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()  # extract the loss value
            if i % 1000 == 999:
                # print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy(model, test_loader)
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % (accuracy))

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel(model)
            best_accuracy = accuracy


def main():
    train_dir = "./data/RubbishClassification/train_.json"
    test_dir = "./data/RubbishClassification/val_.json"
    # get_image(train_dir)
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = TrashSortDataset(train_dir, transform=transformations)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    test_set = TrashSortDataset(test_dir, transform=transformations)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)
    # 打印数据集的长度（总样本数）
    print("Total samples in the dataset:", len(train_set))
    # Instantiate a neural network model
    model = CNN()

    """# 得到一个迭代器
    data_iterator = iter(train_loader)
    # 从迭代器中获取第一个批次的数据
    images, labels = next(data_iterator)
    # 现在，您可以查看第一个批次的图像和标签
    print("Batch of images shape:", images.shape)
    print("Batch of labels:", labels)"""

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    train(model, optimizer, loss_fn, train_loader, test_loader, 5)
    """# 逐个样本遍历数据集并打印前几个样本
    num_samples_to_print = 10
    for i in range(num_samples_to_print):
        image, label = train_data[i]
        print(f"Sample {i + 1} - Label: {label}")
        # 如果您需要显示图像，可以使用以下代码
        cv2.imshow("Image", image.permute(1, 2, 0).numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""


if __name__ == "__main__":
    main()
