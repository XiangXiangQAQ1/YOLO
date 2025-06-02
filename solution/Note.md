# 学习路径
```markdown
# YOLO 论文推荐

在你半月速成、产出项目并写入简历的前提下，**无需啃完所有 YOLO 系列论文**，只需重点关注与 YOLOv5 最相关的两篇核心论文：

---

## 1. YOLOv3（2018）

- **论文标题**：YOLOv3: An Incremental Improvement
- **作者**：Joseph Redmon & Ali Farhadi
- **特色贡献**：
  - 引入 **Darknet-53 backbone**，大幅提升特征提取能力。  
  - 首次结合 **FPN（特征金字塔）** 和 **多尺度预测**，增强对不同尺寸目标的检测能力。
- **阅读价值**：
  - 帮助你快速理解“一阶段检测器”在**速度与精度**上的经典折中。  
  - 与 YOLOv5 的骨干网和多尺度输出设计高度相关。


## 2. YOLOv4（2020）

- **论文标题**：YOLOv4: Optimal Speed and Accuracy of Object Detection
- **作者**：Alexey Bochkovskiy et al.
- **特色贡献**：
  - **数据增强**：Mosaic、Self-Adversarial Training 等技巧强化泛化。  
  - **骨干网**：引入 **CSPDarknet** 结构，提高效率；  
  - **损失函数**：采用 **CIoU Loss**，优化边界框回归准确度。
- **阅读价值**：
  - 大部分改进（如 SPP、C3 模块、超参数策略）在 YOLOv5 源码中沿用。  
  - 理解这些技巧能帮助你在自定义项目中调参和改造模块。

---

## 可选补充（非必须）

- **YOLOv1/v2**：
  - 掌握“一阶段检测”初衷和框架设计，但概念性较强，可选择性阅读。  
- **YOLOv7/v8**：
  - 若后续需要跟进最新迭代，可速读它们在 v5 基础上的新增特性。

---

> 以上两篇论文总页数约 **20 页**，花费 **半天~1 天** 即可完成 `第一遍+重点图表` 的快速阅读，
> 并在阅读后立刻对照 YOLOv5 源码进行实践，保证**理论与工程**同步吸收。
```

| 时间段        | 主要任务                                                                  |
| ------------- | ------------------------------------------------------------------------- |
| **Day 1-4**   | **环境搭建 & 快速跑通示例** <br>- 克隆仓库、安装依赖 <br>- 运行 `detect.py` & `train.py`示例  |
| **Day 5-8**   | **源码顶层架构浏览** <br>- 阅读 `models/common.py`, `models/yolo.py` <br>- 绘制整体数据流图                   |
| **Day 9-12**  | **训练脚本源码剖析** <br>- 深入 `train.py` 主流程 <br>- 改动超参数（如学习率）并对比实验结果              |
| **Day 13-16** | **推理脚本源码剖析** <br>- 深入 `detect.py` <br>- 调整 NMS 阈值 & 输出格式，观察检测效果差异          |
| **Day 17-20** | **项目整合与产出** <br>- 使用自有数据集完整跑通训练与检测 <br>- 产出示例图 & mAP/速度曲线，并撰写简要报告 |



# 准备环境
1. clone 仓库
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

```
# 模型使用

## Train
- 肯定需要训练自己的模型啊？这样才知道你模型中都有什么类
```python
# Train YOLOv5n on COCO128 for 3 epochs
python train.py --data coco128.yaml --epochs 3 --weights yolov5n.pt --batch-size 128

# Train YOLOv5s on COCO for 300 epochs
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5s.yaml --batch-size 64

# Train YOLOv5m on COCO for 300 epochs
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5m.yaml --batch-size 40

# Train YOLOv5l on COCO for 300 epochs
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5l.yaml --batch-size 24

# Train YOLOv5x on COCO for 300 epochs
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5x.yaml --batch-size 16
# 自定义数据集
python train.py --data coco128.yaml --epochs 3 --weights yolov5s.pt --batch-size 128
```

## 推理(Inference)
- 在模型训练和验证调优完成后，用最终的模型权重在新的、未见过的数据（比如测试集或线上业务数据）上做前向预测，得到检测框、类别和置信度。
- 何时使用：当你需要把模型“上线”或生成最终报告时，就调用推理；也可以在验证完成后，用推理脚本批量跑测试集，生成结果文件。
- 不需要test集有label数据，只关注最终带框的结果
1. `PyTorch Hub API`
- 非常简单，几行代码即可。
```python
import torch

# Model loading
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Can be 'yolov5n' - 'yolov5x6', or 'custom'

# Inference on images
img = "https://ultralytics.com/images/zidane.jpg"  # Can be a file, Path, PIL, OpenCV, numpy, or list of images

# Run inference
results = model(img)

# Display results
results.print()  # Other options: .show(), .save(), .crop(), .pandas(), etc. Explore these in the Predict mode documentation.
```


2. 推理，使用训练好的权重和`detect.py`
- 不要加实时显示，服务器上显示不出来
```bash
# 对本地图片
python detect.py --weights yolov5s.pt --source path/to/image.jpg

# 对本地视频并保存结果视频到 runs/detect/
python detect.py --weights yolov5s.pt --source path/to/video.mp4 --save-conf

# 从摄像头实时检测
python detect.py --weights yolov5s.pt --source 0         # 0 表示第一个摄像头设备

# 对 RTSP 流进行检测
python detect.py --weights yolov5s.pt --source 'rtsp://地址/流'

```





## val(valuate)
- 评估模型，在有label的测试集上评估模型，就像trian中的评估结果，不带最终的框
```python
python val.py --weights runs/train/exp5/weights/best.pt --data coco128.yaml --img 640 --half
```

## YOLOV5 输出格式
在你运行 detect.py 完成推理后，YOLOv5 默认会在本地磁盘上生成以下几类输出文件（均位于 runs/detect/exp/ 目录，exp 会自动递增为 exp2, exp3 …）：

1. **带标注的静态图片**
- 格式：.jpg（或根据你输入的源文件格式保持一致）
- 内容：在原图上用彩色矩形框（bounding box）标出每个检测到的目标，并在框上方标注类别名称及置信度（confidence）
- 文件名：和源图同名，例如输入 zidane.jpg → 输出 zidane.jpg，但已加上检测框

2. 可选的文本及裁剪结果（需要额外开启对应参数）
- --save-txt：为每张图片／每段视频生成同名 .txt 文件，里面是所有检测框的坐标（中心点归一化后的 x y w h）及类别索引，方便后续做评估或训练标注

- --save-crop：把每个检测到的目标单独裁剪出来，保存为子图，文件名形如 class_label_confidence.jpg

# 模型增强

## Pruning(剪枝)
python val.py --weights yolov5s.pt --data coco128.yaml --img 640 --half


## 冻结层
- 作用：解决资源，不用计算某些层的权重，就不用反向传播了
- 结果表明，冻结层可以显著加快训练速度，但可能会导致最终mAP（平均精度）略有下降。对所有层进行训练通常可以获得最佳精度，而冻结更多层可以加快训练速度，但代价可能是降低性能。

# 解读输出
一、如何解读 results.png
损失曲线（左图）

box loss：边框回归损失，越低表示预测框与真实框越吻合。

obj loss：对象存在性损失，控制模型判断像素是否含目标。

cls loss：分类损失，当你的数据只有一个类别时这条曲线可能接近 0。

如果损失在不断下降且趋于平稳，说明模型在学习；若出现振荡或不降反升，可能学习率过高或数据有噪声。

指标曲线（中图）

Precision（精确率）：预测正确的框 / 总预测框；

Recall（召回率）：预测正确的框 / 总真实框；

mAP@0.5：IoU ≥ 0.5 时的平均精度；

mAP@0.5:0.95：IoU 从 0.5 到 0.95（步长 0.05）下的平均 mAP。

mAP@0.5(mean Average Precision)
定义使用固定的 IoU 阈值 0.5判断一次检测是否为 TP（True Positive）：
若预测框与某个真实框的 IoU ≥ 0.5，且类别一致，记为 TP；否则为 FP。
在该 IoU 阈值下，针对每个类别分别计算 AP，最后对所有类别取平均，得到 mAP（mean AP）@0.5。

**计算流程**:
匹配：将所有预测框按置信度从高到低排序，依次与尚未被匹配的真实框匹配，判断 IoU ≥ 0.5 即为 TP，否则 FP。

生成 PR 曲线：随着阈值变化（或直接用排序后的置信度序列），记录 Precision 与 Recall。

计算 AP：计算曲线下面积。

平均：对所有类别的 AP 取平均即得到 mAP@0.5。

意义

直观反映模型在“只要 IoU ≥ 0.5”这一松散匹配标准下的整体检测性能。

易理解易对比，但对于定位误差容忍度较高。

3. mAP@0.5:0.95
定义

不仅在单一 IoU 阈值下计算 AP，而是在 多个 IoU 阈值（从 0.50、0.55、0.60 … 到 0.95，间隔 0.05，共 10 个阈值）下分别计算 AP，再对这 10 个 AP 求平均。

对每个 IoU 阈值 t（0.50, 0.55, …, 0.95）分别执行上文 mAP@0.5 那样的匹配、PR 曲线和 AP 计算。对这 10 个 AP 值取平均。

意义
更严格、更全面地评估模型的定位精度。

既考察“宽松匹配”（IoU=0.5）下的检测效果，也考察“严格匹配”（IoU=0.95）下的高精度框选能力。

被 COCO 等主流数据集广泛采用，能够更公平、细粒度地反映模型优劣
这些曲线能直观反映模型在不同阈值下的性能，理想情况是 Precision 和 Recall 同时较高，mAP 稳定上升。

速度曲线（右图）

包括各阶段前向推理（inference）、NMS、后处理等耗时，帮助你评估模型在目标设备上的实时性能。

二、如何解读混淆矩阵