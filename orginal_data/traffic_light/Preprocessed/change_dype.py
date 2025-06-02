import yaml
import os

# 图像分辨率（建议根据你的图片实际尺寸设置）
img_w, img_h = 1280, 720

yaml_path = r"C:\Users\86187\Desktop\college\project\additional_train.yaml"
output_dir = "labels"

# 加载 YAML 数据
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# 🔍 自动提取所有出现的 label 并建立映射
label_set = set()
for item in data:
    for box in item.get("boxes", []):
        if "label" in box:
            label_set.add(box["label"])

label_list = sorted(label_set)
label_map = {label: idx for idx, label in enumerate(label_list)}

print("类别映射表：")
for label, idx in label_map.items():
    print(f"  {label}: {idx}")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 坐标转换函数（XYWH 中心形式）
def to_yolo(x, y, w, h):
    return [(x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h]

# 遍历每一张图片的标注
for item in data:
    name = os.path.splitext(os.path.basename(item['path']))[0]
    label_file = os.path.join(output_dir, name + ".txt")

    with open(label_file, 'w') as f:
        for box in item.get("boxes", []):
            try:
                x_min = box["x_min"]
                y_min = box["y_min"]
                x_max = box["x_max"]
                y_max = box["y_max"]
                w = x_max - x_min
                h = y_max - y_min
                xc, yc, wn, hn = to_yolo(x_min, y_min, w, h)

                label_name = box.get("label", None)
                if label_name not in label_map:
                    print(f"⚠️ 未知类别 {label_name}，跳过")
                    continue

                label_id = label_map[label_name]
                f.write(f"{label_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

            except KeyError as e:
                print(f"⚠️ 缺失字段 {e}，跳过此框")
