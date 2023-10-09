import matplotlib
import torch  # 导入pytorch
from jedi.api.refactoring import inline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from torch import nn, optim  # 导入神经网络与优化器对应的类
import torch.nn.functional as F
from torchvision import datasets, transforms  # 导入数据集与数据预处理的方法
import matplotlib.pyplot as plt


# 定义一个函数用于计算准确率
def calculate_accuracy(model, dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            correct += torch.sum(equals).item()
            total += labels.size(0)
    return correct / total


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)  # 第一个层输入784，输出256
        self.fc2 = nn.Linear(256, 128)  # 第二个层输入256，输出128
        self.fc3 = nn.Linear(128, 64)  # 以此类推
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))  # 之前讲的relu激活函数
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)  # 使用对数的softmax分类

        return x


# 数据预处理和加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.FashionMNIST('dataset/', download=True, train=True, transform=transform)
X = dataset.data.float()
y = dataset.targets

# 定义交叉验证的折数
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# 创建神经网络模型
model = Classifier()

# 定义损失函数和优化器
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# 定义训练时期数
epochs = 15

# 存储准确率
accuracies = []

# 训练和评估模型
for fold, (train_indices, test_indices) in enumerate(k_fold.split(X), 1):
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # 训练模型
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # 评估模型
    with torch.no_grad():
        model.eval()
        output = model(X_test)
        predicted = torch.argmax(output, 1)
        accuracy = accuracy_score(y_test, predicted)
        accuracies.append(accuracy)

    print(f'Fold {fold}: Accuracy = {accuracy:.4f}')

# 计算平均准确率
avg_accuracy = sum(accuracies) / len(accuracies)
print(f'Average Accuracy: {avg_accuracy:.4f}')

'''
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# 下载Fashion-MNIST训练集数据
trainset = datasets.FashionMNIST('dataset/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 下载Fashion-MNIST测试集数据
testset = datasets.FashionMNIST('dataset/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
image, label = next(iter(trainloader))

# 对上面定义的Classifier类进行实例化
model = Classifier()

# 定义损失函数为负对数损失函数，目标是把损失函数降到最低
criterion = nn.NLLLoss()

# 优化方法为Adam梯度下降方法，学习率为0.003
optimizer = optim.Adam(model.parameters(), lr=0.003)

# 对训练集的全部数据学习15遍，这个数字越大，训练时间越长
epochs = 15

# 将每次训练的训练误差和测试误差存储在这两个列表里，后面绘制误差变化折线图用
train_losses, test_losses = [], []

print('开始训练')
for e in range(epochs):
    running_loss = 0

    # 对训练集中的所有图片都过一遍
    for images, labels in trainloader:
        # 将优化器中的求导结果都设为0，否则会在每次反向传播之后叠加之前的
        optimizer.zero_grad()

        # 对64张图片进行推断，计算损失函数，反向传播优化权重，将损失求和
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 每次学完一遍数据集，都进行以下测试操作
    else:
        test_loss = 0
        accuracy = 0
        # 测试的时候不需要开自动求导和反向传播
        # 关闭Dropout
        model.eval()
        with torch.no_grad():

            # 对测试集中的所有图片都过一遍
            for images, labels in testloader:
                # 对传入的测试集图片进行正向推断、计算损失，accuracy为测试集一万张图片中模型预测正确率
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                # 等号右边为每一批64张测试图片中预测正确的占比
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        # 恢复Dropout
        model.train()
        # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        # 输出每轮训练后的准确率
        train_accuracy = calculate_accuracy(model, trainloader)
        test_accuracy = calculate_accuracy(model, testloader)
        print(f'Epoch {e + 1}/{epochs}, '
              f'Training Loss: {running_loss / len(trainloader):.3f}, '
              f'Test Loss: {test_loss / len(testloader):.3f}, '
              f'Training Accuracy: {train_accuracy * 100:.2f}%, '
              f'Test Accuracy: {test_accuracy * 100:.2f}%')

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend()
plt.show()  # 显示绘图
'''
