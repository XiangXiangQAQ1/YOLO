#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File:        detect_test.py
Description: 使用训练好的 YOLOv5 模型对测试集图片进行目标检测，
             并将结果保存到指定目录。
Author:      ChatGPT (OpenAI o4-mini)
Created:     2025-05-13 23:30:00
"""

import argparse
from pathlib import Path
import glob
import cv2
import torch
def detect_test(weights: str,
                source: str,
                output: str,
                imgsz: int = 640,
                conf_thres: float = 0.25,
                iou_thres: float = 0.45):
    """
    使用自定义权重对目标检测并保存结果

    Args:
        weights (str): 模型权重文件路径 (.pt)
        source (str): 测试集图片目录或 glob 模式
        output (str): 检测结果保存目录
        imgsz (int): 输入图片大小
        conf_thres (float): 置信度阈值
        iou_thres (float): NMS IoU 阈值
    """
    # 加载本地模型
    model = torch.hub.load('./', 'custom', path=weights, source='local')
    # model = model.autoshape()
    model.conf = conf_thres
    model.iou = iou_thres

    # 创建输出目录
    save_dir = Path(output)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图片路径（jpg/png/jpeg）
    # 手动读取 source 中所有图片路径
    image_paths = list(Path(source).rglob("*.jpg"))  # 或其他扩展名
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 模型推理
        results = model(img_rgb, size=imgsz)

        # 注意：results.ims[0] 是只读的，需要复制一下
        results.render()
        rendered = results.ims[0].copy()  # 可写副本

        # 转换为 BGR 存盘
        out_img = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

        save_path = Path(output) / img_path.name
        cv2.imwrite(str(save_path), out_img)

    print(f"✅ 检测完成，保存到 {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 YOLOv5 模型对测试集进行目标检测并保存结果")
    parser.add_argument('--weights', type=str, required=True,
                        help='训练好的模型权重文件路径（.pt）')
    parser.add_argument('--source', type=str, default='datasets/coco128/images/train2017',
                        help='测试集图片所在目录或 glob 模式（如 data/test/images）')
    parser.add_argument('--output', type=str, default='runs/detect/test_results',
                        help='检测结果保存目录')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='推理时的输入图片大小')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值 (0–1)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS 时的 IoU 阈值 (0–1)')

    args = parser.parse_args()
    detect_test(args.weights, args.source, args.output, args.imgsz, args.conf, args.iou)
