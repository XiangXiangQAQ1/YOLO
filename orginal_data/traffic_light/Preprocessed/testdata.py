import os
import shutil
import random

# 路径配置
train_img_dir = 'train/images'
train_lbl_dir = 'train/labels'
test_img_dir = 'test/images'
test_lbl_dir = 'test/labels'

# 创建 test 文件夹（如果不存在）
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_lbl_dir, exist_ok=True)

# 获取所有图像文件
image_files = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 设置训练集比例
train_ratio = 0.8  # 80%

# 先把 test 文件夹清空
for f in os.listdir(test_img_dir):
    os.remove(os.path.join(test_img_dir, f))
for f in os.listdir(test_lbl_dir):
    os.remove(os.path.join(test_lbl_dir, f))

# 打乱顺序，随机划分
random.seed(45)  # 保证结果可复现
random.shuffle(image_files)

num_train = int(len(image_files) * train_ratio)
train_set = set(image_files[:num_train])
test_set = set(image_files[num_train:])

# 把 test_set 的图片和标签移走到 test 文件夹
for image_file in test_set:
    # 移动图像
    src_img = os.path.join(train_img_dir, image_file)
    dst_img = os.path.join(test_img_dir, image_file)
    shutil.move(src_img, dst_img)

    # 移动对应标签
    label_file = os.path.splitext(image_file)[0] + '.txt'
    src_lbl = os.path.join(train_lbl_dir, label_file)
    dst_lbl = os.path.join(test_lbl_dir, label_file)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

print(f"划分完成：训练集 {len(train_set)} 张，测试集 {len(test_set)} 张。")
