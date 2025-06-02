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


✅ 第一阶段：跑通基础流程 + 理解训练流程（1~1.5周）
🎯 目标
跑通 YOLOv5 训练 & 推理流程

明白每个主要文件的职责和调用逻辑

📌 任务清单
搭环境 + 跑通推理

clone YOLOv5 官方项目

下载模型权重 yolov5s.pt

用 detect.py 推理几张图片（包括你自己的）

跑训练流程

使用小数据集（如 VOC、COCO 子集或自定义 10 张图）训练

命令：python train.py --img 640 --batch 16 --epochs 10 --data yourdata.yaml --weights yolov5s.pt

理解主线代码结构

梳理调用关系（建议你画图或做笔记）：

markdown
复制
train.py → model.py → common.py
           ↓
         loss.py
           ↓
         datasets.py
每看一个文件，只看主函数 + 函数调用结构，不要深挖细节

调试模型

加断点（如在 loss 处打印各项损失）

修改训练轮数、图片大小、backbone 网络（如切换 yolov5s.yaml 为 yolov5m.yaml）

✅ 第二阶段：理解模型结构 + 算法原理（1~2周）
🎯 目标
理解 YOLOv5 的模型组成（backbone、neck、head）

理解核心算法：anchor、IoU、NMS、损失函数等

📌 任务清单
阅读结构定义

重点阅读：

models/yolov5s.yaml（网络结构）

models/common.py（每个模块如 C3、Focus）

你已经看过 models.py，可以回头结合这些理解

学会从 .yaml 文件到实际模型构建的过程

深入理解损失函数

阅读 loss.py

搞懂三种损失：

位置（bbox）损失

类别损失

目标置信度损失

参考博客/视频补课，如：

https://zhuanlan.zhihu.com/p/359246748

学习YOLO基本原理（非v5专属）

看 YOLOv3 论文或解读视频

理解 anchor 机制、grid cell 思想、IoU计算等

✅ 第三阶段：尝试改进 + 应用开发（2~3周）
🎯 目标
自定义模型结构并训练

尝试调参、剪枝、加CBAM、改loss等

做实际项目或小Demo

📌 任务清单
自定义模型结构

改 .yaml 文件添加模块（如注意力机制）

加 CBAM、SE 或 GhostConv 模块（建议从 common.py 中找模板）

调参实验

修改 anchor、img-size、batch size、训练策略

比较训练/验证结果，学会看 PR 曲线、mAP

项目实践

用自己数据做一个目标检测 demo（如检测猫狗、烟头、货物等）

导出模型，部署在 Flask/Web 或 ONNX/TensorRT

🔍 补充建议
可选论文参考：

YOLOv3：https://pjreddie.com/media/files/papers/YOLOv3.pdf

YOLOv4：https://arxiv.org/abs/2004.10934

YOLOv7/YOLOv8 可做对比阅读（后期）

调试建议：

用 VSCode 的 Debug 模式，或插入 print() 看 shape / loss

多画网络结构图（比如用 Netron 看 pt 文件）

遇到问题怎么办？



----------------------
# 第一步：建立“结构图 + 流程图 + 数据流”三件套
你可以用纸或白板/画图工具，把 YOLOv5 的整个训练过程，画出如下内容：

图类型	你要标注的东西
模块结构图	模块之间连接关系（Backbone → Neck → Head）
数据流图	每一层 feature map 的 shape（例如 [B, 256, 40, 40]）
控制流图	train.py 中的调用顺序：model → optimizer → loss → backprop

## 尝试回答这些问题：

“train() 中调用了 model.forward()，那具体是哪个 forward？”

“Loss 是在哪里定义？objectness 是怎么算的？anchor 是怎么匹配的？”

“Detect.forward() 是怎么从 feature map 变成坐标和分类的？”

#  第二步：带目标精读 + Debug
挑选一个细节问题，比如：

YOLOv5 是怎么把 anchor 和 GT 匹配的？IOU 是怎么算的？

你去代码中精读这部分，比如 compute_loss() 中：

python
复制代码
build_targets() 会做 anchor 与 gt 的匹配
📌 然后你加 log/断点调试：

每个 anchor 是怎么选中与哪个 gt 配对的；

objectness 是在哪里赋值的；

为什么有 ignore 区域？

🔁 对每一个细节做“运行级理解”，比单纯读 5 遍代码都有效。

🧪 第三步：动手改一点点东西
你可以做一个微小但目标明确的改动，比如：

改动	目的
修改 Detect 输出的维度/anchor 数量	理解 YOLOv5 的输出结构
改一个 loss 计算方式，比如加 focal loss	理解 loss 的组成
改 forward 的一部分，加入 CBAM/注意力模块	理解模型结构构建流程

📌 小改动 + 小实验，迫使你彻底理解“模型 forward 是怎么走的、哪里会出错、输出维度怎么控制”。

💡 关键点总结：
阶段	做法
📖 看懂代码	多读几遍，画图总结，理解结构
🧠 真理解	跑通代码 + 逐行追踪 + 拆解细节
🛠 掌握	自己动手改、跑实验，具备 Debug 能力

你已经在第 1 阶段做得不错了，只要你从现在开始认真做第 2 和第 3 步，很快就会从“能看懂”提升到“能掌握”，甚至能自己写 YOLO Head/替换
-----------




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
python train.py --data coco128.yaml --epochs 3 --weights yolov5s.pt --batch-size 128

# Train YOLOv5s on COCO for 300 epochs
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5s.yaml --batch-size 64

# Train YOLOv5m on COCO for 300 epochs
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5m.yaml --batch-size 40

# Train YOLOv5l on COCO for 300 epochs
python train.py --img 640 --data coco.yaml --epochs 300 --weights '' --cfg yolov5l.yaml --batch-size 24

# Train YOLOv5x on COCO for 300 epochs
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5x.yaml --batch-size 16
# 自定义数据集
python train.py --data coco128.yaml --epochs 3 --weights yolov5s.pt --batch-size 128

python train.py --data traffic_light.yaml --epochs 3 --weights yolov5s.pt --batch-size 128
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


python detect.py --img 640 --weights /root/dip_do/solution/yolov5/runs_saved/train/Train_for_coco128/weights/best.pt --source '/root/dip_do/solution/yolov5/datasets/coco128/images/train2017'

python detect.py --img 640 --weights /root/dip_do/solution/yolov5/runs_saved/train/traffic_light_v.1.1/weights/best.pt --source '/root/dip_do/orginal_data/traffic_light/test/images'

```





## val(valuate)
- 评估模型，在有label的测试集上评估模型，就像trian中的评估结果，不带最终的框
```python
python val.py --weights runs/train/exp5/weights/best.pt --data coco128.yaml --img 640 --half

python val.py --weights /root/dip_do/solution/yolov5/runs_saved/train/traffic_light_v.1.1/weights/best.pt --data traffic_light.yaml --img 640 --half

# 用 test 集评估 mAP
python val.py --weights runs/train/exp/weights/best.pt \
              --data data.yaml \
              --task test

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
先检验一下不剪枝的效果(half 用于加速)
```python
python val.py --weights pretrained/yolov5m.pt --data coco128.yaml --img 640 --half
```



## 冻结层
- 作用：解决资源，不用计算某些层的权重，就不用反向传播了
- 结果表明，冻结层可以显著加快训练速度，但可能会导致最终mAP（平均精度）略有下降。对所有层进行训练通常可以获得最佳精度，而冻结更多层可以加快训练速度，但代价可能是降低性能。冻结前十层观测层很常见。
```python
python train.py  --epochs 100 --weights /root/dip_do/solution/yolov5/pretrained/yolov5s.pt --data coco128.yaml --batch-size 128 --freeze 10
```
```python
python train.py \
  --data coco128.yaml \
  --weights /root/dip_do/solution/yolov5/pretrained/yolov5m.pt\               # 加载预训练 coco 模型
  --cfg models/yolov5m.yaml \          # 明确结构中的 nc=你自己的类别数
  --epochs 100 --img 640 --batch-size 16

```
## 


# 解读输出
## Train 输出

### 如何解读 results.png
损失曲线（左图）

- box loss：边框回归损失，越低表示预测框与真实框越吻合。
- obj loss：对象存在性损失，控制模型判断像素是否含目标
- cls loss：分类损失，当你的数据只有一个类别时这条曲线可能接近 0。
如果损失在不断下降且趋于平稳，说明模型在学习；若出现振荡或不降反升，可能学习率过高或数据有噪声。

指标曲线（中图）

- Precision（精确率）：预测正确的框 / 总预测框；
- Recall（召回率）：预测正确的框 / 总真实框；表示在所有实际为正的样本中，模型找对了多少。
- mAP@0.5：IoU ≥ 0.5 时的平均精度；
- mAP@0.5:0.95：IoU 从 0.5 到 0.95（步长 0.05）下的平均 mAP。

mAP@0.5(mean Average Precision)
定义使用固定的 IoU 阈值 0.5判断一次检测是否为 TP（True Positive）：
若预测框与某个真实框的 IoU ≥ 0.5，且类别一致，记为 TP；否则为 FP。
在该 IoU 阈值下，针对每个类别分别计算 AP，最后对所有类别取平均，得到 mAP（mean AP）@0.5。

### 判别输出
- $mAP@0.5 >0.75$: 为优秀
- $mAP@0.5 \in (0.60,0.75]$: 为合格
- mAP@0.5:0.95 要比mAP 低0.15~0.20
**计算流程**:
- 匹配：将所有预测框按置信度从高到低排序，依次与尚未被匹配的真实框匹配，判断 IoU ≥ 0.5 即为 TP，否则 FP。
- 生成 PR 曲线：随着阈值变化（或直接用排序后的置信度序列），记录 Precision 与 Recall。计算 AP：计算曲线下面积。
- 平均：对所有类别的 AP 取平均即得到 mAP@0.5。

- 意义：直观反映模型在“只要 IoU ≥ 0.5”这一松散匹配标准下的整体检测性能。

3. mAP@0.5:0.95
- 定义：不仅在单一 IoU 阈值下计算 AP，而是在 多个 IoU 阈值（从 0.50、0.55、0.60 … 到 0.95，间隔 0.05，共 10 个阈值）下分别计算 AP，再对这 10 个 AP 求平均。对每个 IoU 阈值 t（0.50, 0.55, …, 0.95）分别执行上文 mAP@0.5 那样的匹配、PR 曲线和 AP 计算。对这 10 个 AP 值取平均。

- 意义：更严格、更全面地评估模型的定位精度。既考察“宽松匹配”（IoU=0.5）下的检测效果，也考察“严格匹配”（IoU=0.95）下的高精度框选能力。

被 COCO 等主流数据集广泛采用，能够更公平、细粒度地反映模型优劣，这些曲线能直观反映模型在不同阈值下的性能，理想情况是 Precision 和 Recall 同时较高，mAP 稳定上升。

## 曲线

### Confusion_matrix
- X轴（横轴）：表示真实的标签（True Labels）。
- Y轴（纵轴）：表示模型预测的标签（Predicted Labels）。
- 每个格子的颜色表示该类在预测中的比例，颜色越深表示准确率越高（最多为1.0）。
- 右侧的颜色条是比例颜色图例（Colorbar），0 表示没有预测为该类，1 表示预测全对。

1. 对角线（从左上到右下）
- 理想情况下，所有预测应该集中在对角线上，表示模型预测与真实标签完全一致。
- 每个对角线上较深的蓝色格子说明模型在该类上预测得非常准确。

2. 对角线之外的格子
- 表示模型将一个类别错误地预测成了另一个类别。
- 比如：如果在横轴上的“dog”对应纵轴上的“cat”处有较深的颜色，说明模型经常把“dog”预测成“cat”。


###  Precision-Confidence Curve（精度-置信度曲线）
**横轴 (X-axis): Confidence**
- 表示模型对预测结果的“信心”，取值范围是 0 到 1。如果样本预测的概率大于该threshold value, 则判断为正值，否则判断为负值
- 置信度越高，表示模型越“确信”这个预测是正确的。

**纵轴 (Y-axis): Precision（精度）**
- 精度 = TP / (TP + FP)，
- 表示在所有预测为正样本中，真正为正的比例。要求不能把负样本预测为正样本，可以用于垃圾邮件的检测，不能把正常邮件当垃圾邮件处理。
- 当置信度越高时，预测正样本的准度越高，所以精度越高。
`all classes 1.00 at 0.959`:表示当置信度阈值设置为 0.959 时，平均精度达到 1.00.

**你能从图中看出什么？**
高性能模型：
- 蓝线（平均精度）在整个置信度区间内都比较高，说明模型整体表现不错。
- 最佳精度在置信度接近 1 的时候达到最大，意味着模型在“非常自信”的时候几乎不会犯错。

**实际应用中该怎么看？**
- 如果你想部署模型，通常会根据这个图来选择一个“合适的置信度阈值”（比如 0.5 或 0.6），以在精度和召回之间取得平衡。
- 你可以使用 0.95 作为阈值来获得非常高的精度，但这可能会牺牲召回率（漏检多）。

### F1-Confidence Curve
- 纵坐标为$F_1 = 2\frac{Precision*Recall}{Precision+Recall}$，期望P和R同时高
- 其余部分和P_curve 一样

### R-confidence
- 纵坐标为Recall, 其余和P_curve 一样
- Recall = TP/(TP+FN), 关注点为真正为正的样本，要求不能把真正为正的样本预测为负样本。可以用于医院癌症的检测。
- 置信度越高， 模型越胆小，越可能把正样本预测为负样本。
## Valuate 输出



## 断点
在断点处暂停后，你可以通过调试工具单步执行：

F10：执行当前行并跳到下一行。

F11：进入函数（比如 compute_loss()）。

Shift+F5：停止调试。

如果你想跳过当前的循环迭代，而不直接跳出循环，可以在调试控制台输入：`
next`