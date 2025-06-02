import os
import shutil

# 灰度图像目录
gray_image_dir = r'additional\output_images'
# 所有原始标签的总文件夹
all_labels_dir = r'labels'
# 输出标签保存位置
output_label_dir = 'output_labels'

os.makedirs(output_label_dir, exist_ok=True)

# 遍历灰度图像（带G后缀的图像）
for filename in os.listdir(gray_image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        name_with_g, _ = os.path.splitext(filename)
        
        if name_with_g.endswith('G'):
            original_name = name_with_g[:-1]  # 去掉末尾的 G 得到原名
            original_label_file = os.path.join(all_labels_dir, original_name + '.txt')
            new_label_file = os.path.join(output_label_dir, name_with_g + '.txt')
            
            if os.path.exists(original_label_file):
                shutil.copy(original_label_file, new_label_file)
                print(f"已复制标签: {original_label_file} -> {new_label_file}")
            else:
                print(f"未找到标签: {original_label_file}")
