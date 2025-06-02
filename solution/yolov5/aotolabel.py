from ultralytics import YOLO
import os

# 1) 加载预训练模型
model = YOLO('pretrained/yolov8n.pt')  # 较小的 nano 版本，速度快

# 2) 批量预测并保存
#    save_txt=True 会在每张图片同名 .txt 中输出 yolo 格式 bbox
results = model.predict(
    source='/root/dip_do/dog_archive/valid/Afghan',  # 你的图片文件夹
    save=True,
    save_txt=True,
    project='auto_labels',  # 输出目录
    name='run',            # 子文件夹名
    exist_ok=True
)
