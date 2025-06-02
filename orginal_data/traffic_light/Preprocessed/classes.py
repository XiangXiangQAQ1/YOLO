import yaml
import os

# 路径配置：确保指向你的 YAML 文件
yaml_path = r"C:\Users\86187\Desktop\college\project\additional_train.yaml"
output_path = "classes.txt"  # 输出的 classes.txt 路径

# 加载 YAML 数据
with open(yaml_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

# 提取所有出现过的 label
label_set = set()
for item in data:
    for box in item.get("boxes", []):
        if "label" in box:
            label_set.add(box["label"])

# 排序并编号，确保顺序一致
label_list = sorted(label_set)  # 字母排序，和前面脚本中一致

# 写入 classes.txt
with open(output_path, 'w', encoding='utf-8') as f:
    for cls in label_list:
        f.write(cls + '\n')

print(f"✅ 已生成 classes.txt，共 {len(label_list)} 个类别：")
for i, cls in enumerate(label_list):
    print(f"  {i}: {cls}")
