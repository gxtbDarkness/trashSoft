# trashSoft

​	本项目通过使用传统机器学习方法（如特征提取 & `SVM`分类器 等）和深度学习方法实现了一个面向垃圾图像回收的图像分类器。以下介绍代码的运行方式。

## 1. 传统机器学习方法

​	在控制台终端通过以下命令行代码运行：

```sh
python main.py <method> -classifier <classifier> -c class1 class2 ... -k <number_of_clusters> -s <number_of_splits> -cval <True/False>
```

​	以下是命令行参数的含义：

- **方法 (`method`)：** 指定要使用的特征提取方法。可选项包括：
  - `SIFT`：尺度不变特征变换（Scale-Invariant Feature Transform）
  - `HOG`：方向梯度直方图（Histogram of Oriented Gradients）
  - `LBP`：局部二值模式（Local Binary Patterns）
- **分类器 (`-classifier`)：** 确定用于分类的机器学习模型。可选项包括：
  - `SVM`：支持向量机（Support Vector Machine）
  - `LOG`：逻辑回归（Logistic Regression）
  - `MLP`：多层感知器分类器（Multi-Layer Perceptron Classifier）
- **类别 (`-c` 或 `--classes`)：** 必需参数，指定用于训练的图像类别。可以作为参数提供多个类别名称。
- **聚类数目 (`-k` 或 `--k`)：** 可选参数，用于SIFT方法。定义聚类的数量。默认值为200.
- **分割数目 (`-s` 或 `--splits`)：** 可选参数，确定KFold交叉验证的分割数目。默认值为5。
- **交叉验证 (`-cval` 或 `--crossval`)：** 可选布尔参数。将其设置为`True`启用KFold交叉验证。默认为`True`。

​	例如，你可以运行以下命令：

```sh
python main.py HOG -classifier SVM -c 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 -k 200 -s 5 -cval True
```

## 2. 深度学习方法

​	在控制台终端通过以下命令行代码运行：

```sh
python cnn.py <model> -pretrained <True/False> -extensive_dataset <True/False> -batch_size <batch_size> -learning_rate <lr> -epochs <ep> -classes_num <cla>
```

以下是命令行参数的说明：

- **模型 (`model`):** 选择要使用的模型，可选项包括：
  - `AlexNet`
  - `ResNet`
- **是否预训练 (`-pretrained`):** 可选布尔参数，默认为`True`，决定是否使用预训练的模型。
- **数据集选择 (`-extensive_dataset`):** 可选布尔参数，默认为`False`，用于选择是否使用扩展的数据集。
- **批量大小 (`-batch_size`):** 可选参数，定义每个批次中样本的数量，默认为32。
- **学习率 (`-learning_rate`):** 可选参数，设置优化器的学习率，默认为0.0001。
- **训练周期 (`-epochs`):** 可选参数，定义模型训练的周期数，默认为100。
- **类别数量 (`-classes_num`):** 可选参数，用于指定数据集中的类别数量，默认为16。

​	例如，你可以运行以下命令：

```sh
python cnn.py AlexNet -pretrained True -extensive_dataset True -batch_size 32 -learning_rate 0.0001 -epochs 50 -classes_num 16
```

## 3. 预测图像标签

​	在控制台终端通过以下命令行代码运行：

```shell
python predict.py <model> -model_path <model_file_path>
```

以下是命令行参数的说明：

- **模型 (`model`):** 选择要使用的模型，可选项包括：
  - `AlexNet`
  - `ResNet`
- **模型路径 (`-model_path` 或 `--path`):** 指定要加载的模型的文件路径，这是一个必需的参数。

注意，`predict.py`代码只接受预训练后的`ResNet`模型或`AlexNet`模型。





