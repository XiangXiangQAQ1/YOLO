import os
import shutil

# 设置图片和标签的根目录
image_root = 'train/images'
label_root = 'train/labels'

def merge_subfolders_to_root(root_path):
    for subfolder in os.listdir(root_path):
        subfolder_path = os.path.join(root_path, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                src_file = os.path.join(subfolder_path, filename)
                dst_file = os.path.join(root_path, filename)
                shutil.move(src_file, dst_file)  # 移动文件
            os.rmdir(subfolder_path)  # 删除空子文件夹

# 合并所有 images 和 labels 的子文件夹内容
merge_subfolders_to_root(image_root)
merge_subfolders_to_root(label_root)

print("✅ 所有子文件夹内容已合并到 train/images 和 train/labels 根目录，并删除了子文件夹。")
