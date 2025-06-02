import os
import shutil
import random

# 原始训练集路径
train_img_dir = 'train/images'
train_lbl_dir = 'train/labels'

# 验证集目标路径
val_img_dir = 'val/images'
val_lbl_dir = 'val/labels'

# 创建 val 目录
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# 获取训练集中所有图像文件名
image_files = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 随机挑选 10 张
random.seed(123)  # 保证可复现
val_images = random.sample(image_files, 10)

# 移动图像和对应标签到 val/
for image_file in val_images:
    # 图像
    shutil.move(
        os.path.join(train_img_dir, image_file),
        os.path.join(val_img_dir, image_file)
    )

    # 标签
    label_file = os.path.splitext(image_file)[0] + '.txt'
    src_lbl = os.path.join(train_lbl_dir, label_file)
    dst_lbl = os.path.join(val_lbl_dir, label_file)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)

print(f"✅ 已从训练集中挑选 10 张图像移动到 val/ 作为验证集。")
