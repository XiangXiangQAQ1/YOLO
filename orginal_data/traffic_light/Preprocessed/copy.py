import os
import shutil

src_dir = r'output_labels'
dst_dir = r'labels'

os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    if filename.endswith('.txt'):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy(src_path, dst_path)
        print(f"已复制: {src_path} -> {dst_path}")
