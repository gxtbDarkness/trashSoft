from PIL import Image
import os
import json
import random
from pyecharts.charts import Bar
from pyecharts import options as opts


def is_image_file(filename):
    # 判断文件扩展名是否为常见的图片格式
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    return any(filename.lower().endswith(ext) for ext in image_extensions)


def check_image_integrity(file_path):
    try:
        # 尝试打开图像文件
        with Image.open(file_path) as img:
            # 获取图像的基本信息，例如尺寸、模式等
            img_info = img.info
            width, height = img.size
            mode = img.mode

            # 在这里可以添加其他关于图像完整性的检查

            print(f"图像文件: {file_path}, 尺寸: {width} x {height}, 模式: {mode}, 信息: {img_info}")

    except Exception as e:
        # 图像文件损坏或格式不正确时，捕获异常并输出错误信息
        print(f"错误：{file_path} - {e}")


def check_images_in_folder(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 过滤出图片文件
    image_files = [file for file in files if is_image_file(file)]

    # 输出图片文件列表
    print("图片文件列表:")
    for image_file in image_files:
        print(os.path.join(folder_path, image_file))

    # 检查每个图片文件的完整性
    for image_file in image_files:
        file_path = os.path.join(folder_path, image_file)
        check_image_integrity(file_path)


def rename_images_in_folder(folder_path, prefix="image_", start_index=1):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 过滤出图片文件
    image_files = [file for file in files if is_image_file(file)]

    # 重命名每个图片文件
    index = start_index
    for image_file in image_files:
        old_path = os.path.join(folder_path, image_file)
        # 构建新的文件名
        new_name = f"{prefix}{index:06d}{os.path.splitext(image_file)[1]}"  # 格式化为六位数字
        new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)

        print(f"重命名成功: {old_path} -> {new_path}")

        # 增加索引
        index += 1


# 判断文件扩展名是否为常见的图片格式
def is_image_file(filename):
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    return any(filename.lower().endswith(ext) for ext in image_extensions)


alex_no = {"0": 0.38016528925619836, "1": 0.2608695652173913, "2": 0.5223214285714286, "3": 0.5724907063197026,
           "4": 0.48299319727891155, "5": 0.25, "6": 0.3870967741935484, "7": 0.25, "8": 0.2222222222222222,
           "9": 0.5568181818181818, "10": 0.5692307692307692, "11": 0.32142857142857145, "12": 0.4606741573033708,
           "13": 0.5094339622641509, "14": 0.6491228070175439, "15": 0.7019230769230769}

res_no = {"0": 0.4214876033057851, "1": 0.10144927536231885, "2": 0.5625, "3": 0.6394052044609665,
          "4": 0.35374149659863946,
          "5": 0.20588235294117646, "6": 0.3225806451612903, "7": 0.39285714285714285, "8": 0.18518518518518517,
          "9": 0.7045454545454546, "10": 0.6, "11": 0.21428571428571427, "12": 0.3707865168539326,
          "13": 0.4088050314465409, "14": 0.8771929824561403, "15": 0.8173076923076923}

alex = {"0": 0.8125, "1": 0.7045454545454546, "2": 0.8095238095238095, "3": 0.7291666666666666, "4": 0.95, "5": 0.68,
        "6": 0.7676056338028169, "7": 0.9145299145299145, "8": 0.7132867132867133, "9": 0.8253968253968254,
        "10": 0.8742857142857143, "11": 0.7222222222222222, "12": 0.7368421052631579, "13": 0.7723577235772358,
        "14": 0.8661971830985915, "15": 0.9383561643835616}

res = {"0": 0.9296875, "1": 0.8409090909090909, "2": 0.8367346938775511, "3": 0.8958333333333334,
       "4": 0.9083333333333333, "5": 0.76, "6": 0.8943661971830986, "7": 0.9230769230769231, "8": 0.8671328671328671,
       "9": 0.8333333333333334, "10": 0.8857142857142857, "11": 0.8650793650793651, "12": 0.9385964912280702,
       "13": 0.7723577235772358, "14": 0.9577464788732394, "15": 0.9726027397260274}

classes = ["旧书0", "易拉罐1", "铁丝类2", "废旧鞋子3", "废旧包4", "砖瓦陶瓷5", "果壳瓜皮6", "废弃电池7", "废旧泡沫8",
           "食物残渣9", "硬纸板10", "废纸巾11",
           "塑料袋12", "塑料瓶13", "玻璃瓶14", "碎玻璃15"]

# 创建柱状图
bar = (
    Bar(init_opts=opts.InitOpts(width="1500px", height="800px"))
    .add_xaxis(classes)
    .add_yaxis("Alex_No", [round(value, 2) for value in alex_no.values()])
    .add_yaxis("Res_No", [round(value, 2) for value in res_no.values()])
    .add_yaxis("Alex", [round(value, 2) for value in alex.values()])
    .add_yaxis("Res", [round(value, 2) for value in res.values()])
    .set_global_opts(
        title_opts=opts.TitleOpts(title="不同类别的分类准确率"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
    )
    .render("bar_chart.html")
)



"""# 设置数据文件夹路径
data_folder = './data/expand_data/trainval_'

# 定义训练集和验证集的比例
train_ratio = 0.8
val_ratio = 0.2

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
    json.dump(val_data, val_file)"""

"""# 用法示例
folder_path = "../expand_data/trainval_/15"
rename_images_in_folder(folder_path, prefix="image_", start_index=10297)"""

"""# 用法示例
folder_path = "../expand_data/trainval/1"
check_images_in_folder(folder_path)"""
