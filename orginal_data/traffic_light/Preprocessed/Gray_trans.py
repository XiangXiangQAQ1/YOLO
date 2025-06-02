import cv2
import os

input_dir = r'additional\2015-10-05-14-40-46_bag'
output_dir = 'output_images'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {filename}")
            continue
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 分离文件名和扩展名
        name, ext = os.path.splitext(filename)
        new_name = f"{name}G{ext}"  # 文件名后加G
        
        output_path = os.path.join(output_dir, new_name)
        cv2.imwrite(output_path, gray_img)
        print(f"已保存灰度图: {output_path}")
