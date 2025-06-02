# YOLOV5 model总类

## Conv
- conv: 老朋友，卷积层了，不是人家写的是真高级啊，好好学学,学习的是各种卷积核。输入是图像，一般为彩色图像，有三个通道。模型就会自动创建 64 个「3×3×3 的卷积核」来参与学习，每个卷积核都能学会提取不同的图像特征！
- 但是咱们使用的是特征图，像`x = torch.relu(self.bn1(self.conv1(x))) # [B,24,32,32]`使用的就是特征图，最终可以把所有特征图平展 成一维，用于线性层连接。
- 就是我们前向传播的时候用的是特征图，反向传播更新的是卷积核
```python
conv = nn.Conv2d(3, 64, kernel_size=3)
```

- self.bn = nn.BatchNorm2d(c2),归一化，减轻梯度消失和梯度爆炸
- 这if else 直接给我整不会了。
```pyhton
self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
```

- Conv是一个全连接层，每个输入层和输出层都需要学习一个卷积核，参数的总数为$c_{in} *c_{out}*k_h*k_w$, $k_h$和$k_w$为高和宽

### DWConv Depthwise Convolution（逐通道卷积）
- 使用父类的init 函数，继承父类
- 深度分离卷积(分组卷积),我们将卷积层分为不同的组，而每个组进行独立的卷积，这样组间就没有相互连接，可以有效提高计算效率，减轻计算量。nn.Conv2d中g即为分组数。
    - 且分组是严格按照顺序进行的。


```python
self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False) 
```
```python

class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
```


### Tranform(FFN)
- “向前向后查找”的是 Attention 部分，它通过对 Q/K/V 做加权平均，让每个输出位置都能够访问序列的所有位置。
- FFN 仅仅起到“对已经融合上下文的表示”做逐位置非线性变换的作用，它不负责“查找”或“聚合”不同位置的信息。
- 换句话说，在 Transformer 里，每一层都会先做一次 Attention（获取上下文），然后紧跟一个 FFN（单位置非线性）。只有把两者串在一起，才既保证了“序列内部任意位置的交互”，又补充了足够的非线性表达能力。


### Bottleneck(瓶颈)
- 先压缩再扩张，减少计算量
    - 使用过渡卷积层可以减少参数数量，本来学习(3*3*c1*c2)的卷积核,现在学习(c1*c2/2 + 3*3*c2/2*c2), 若$c1=c2$，则大约减少了一半的计算成本
- 残差连接可以解决梯度消失的问题，因为在反向过程中至少有一个梯度1。使得深层网络的梯度消失问题得到缓解，并加速收敛。
- 当然，这只能解决一层的梯度消失，所以使用n层的话，就形成了全局更稳定的梯度通道，比较好的解决了梯度消失的问题。
```python
   def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
``` 
```python
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1) # kernel =1
        self.cv2 = Conv(c_, c2, 3, 1, g=g) # kernel =3
        self.add = shortcut and c1 == c2

```
- 什么是反向传播：就是用链式法则，把对x的求导从最后一层传到前面来，且用最速下降法途中的各个层的权重

#### BottleneckCSP(Cross Stage Partial Network)
- 它是一个卷积神经网络里的模块，用于处理图片的特征图（feature map），让特征图变得更深、更丰富，但计算量更小、更稳定。
- 首先，cv1和cv2都是学习大小为1的核；且通道数相对于输出减半。
- 之后左边主干路学习n层的Bottlneck层，学习复杂的特征向` self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))`
- 右边shortcut路为恒等变换，保留了图片的浅层特征。
- 最后再用`cat`，将两层拼接起来，(根据通道数拼接)
```bash
          x
         / \
        |   |
      cv1   cv2
       |     |
      m()   identity
       |     |
      cv3   |
        \   /
        cat(y1, y2)
           |
          BN
           |
          SiLU
           |
          cv4
           ↓
         Output
```
```python
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1)))) # 表示在通道数拼接。
```


### cross convolution(交叉卷积下采样)
- 现在我们的卷积核为1维的，且一个输入层对应一个输出层。称为(Depthwise)。
```md
假设你有一个输入张量：[C_in=3, H=32, W=32];
你用一个 3×3 的 depthwise 卷积：
- 你有 3 个卷积核（每通道一个）；
- 每个卷积核是 3×3，只作用于 自己的通道；
- 输出仍然是 3 个通道，不会混合通道之间的信息。
```
- cross convolution 通过两个方向的交叉卷积来提取特征(1,k),(k,1)


## C3( CSP Bottleneck with 3 Convolutions)
- C3 是 YOLOv5 的优化版 CSP 模块，更高效;BottleneckCSP 是原始定义的 CSP，更严谨
- 且C3的conv是再ultralyics 定义的，而不是原始的`nn.Conv2d`
```bash
          x
         / \
        |   |
      cv1   cv2
       |     |
      m()  identity
       |     |
        \   /
        cat(y1, y2)
             |
            cv3
             ↓
          Output

```

```python
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

```

1. **c3x**
将bottlenck替换成CrossConv
```python
self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))
```

2. **C3TR**
- 把 Bottleneck 替换成了Transformer Block。

3. **C3SPP**
- 替换为 SPP 模块（空间金字塔池化），用不同尺寸的 max pooling 来处理图像

4. **C3Ghost**
- 使用 GhostNet 的思想：把一部分特征用 cheap 操作（如 depthwise conv）来生成。
```python 
self.m = nn.Sequential(*(GhostBottleneck(...) for _ in range(n)))
s
```


## SPP(Spatial Pyramid Pooling)
- 使用三个不同的maxpool层提取信息，最后整合三个池化层，再通过`cv2`卷积。
- Pyramid（金字塔）：指的是它使用了多种不同尺度的池化核，比如 5×5、9×9、13×13，这些大小就像一个金字塔的不同“层级”，感受野从小到大
```md
               x
               |
             Conv1x1 (降通道)
               |
         ┌─────┼──────┬──────┬──────┐
         │     │      │      │      │
        ID   Max5   Max9   Max13  （分别对应 kernel_size）
         │     │      │      │
         └─────┴──────┴──────┘
               │
            Concat
               │
             Conv1x1 (升通道)
               ↓
            输出特征图

```

```python
class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
```

- 优点：通过不同大小的池化核提取局部和全局信息；且有助于检测不同大小的目标，MaxPool 是无参数操作，不增加模型容量

### Maxpool
- maxpool通过提取指定区域的最大值来提取特征，kernel_size=5，表示每次用5*5的框提取其中的最大值。
- padding用于补零，保持图片尺寸不变。
```python
nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
```


### SPPF(Spatial Pyramid Pooling - Fast )
- 用更少的计算获得类似的多尺度感受野效果.使用5*5的maxpool的嵌套。获得了和SPP一样5*5,9*9,13*13同样的结果
    - 因为我们使用的stride=1，每次移动1，相当于扩大一层，而5*5，去除本身的一层，刚好要扩大四层

```python
self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

 y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
```



## Focus
- 把空间信息（宽、高）压缩到通道维度中，从而减小空间尺寸的同时保留更多细节,并且保留了信息，不像池化一样会丢失信息。
```python
x[..., ::2, ::2] → 从偶数行偶数列取像素
x[..., 1::2, ::2] → 奇数行偶数列
x[..., ::2, 1::2] → 偶数行奇数列
x[..., 1::2, 1::2] → 奇数行奇数列
```
这样就从原图像中提取了 4 个子图，每个大小是 [B, C, H/2, W/2]
- 假设输入图像是 640x640 的 RGB 图片 [1, 3, 640, 640]，经过 Focus 后：输出维度是 [1, 12, 320, 320]（C=3，压缩空间维度到通道上，4C=12）
    - 空间尺寸减半，但内容没丢失（只是重新排列了），便于后续操作。

- Focus 是 YOLOv5 输入阶段的第一步。它通常后接一个普通的卷积层做特征提取。例如

## Ghost
### GhostConv
- 卷积学习的代价很昂贵，所以我们先主要学习一部分特征，另一部分就先当幽灵。`gruop =c_`

```python
class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)// group =c_

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)
```

### Ghost Bottleneck
- 使用Ghost conv的 Bottenlenck;依然是先压缩再升维；但有两点区别，它对于s==2单独处理了。GhostNet 的设计哲学一致：主分支生成主特征，Ghost 分支生成 cheap 特征，下采样用正统轻量卷积处理。
- Ghost conv的主要任务为提取特征，而不是处理空间结构，对它进行下采样，会让本来简单的结构变得复杂；
- 而且使用DWconv进行下采样也非常轻量。高效
```bash
+--------------------+
|      Input         |
+--------------------+
          |
          v
+--------------------+
|   GhostConv (1x1)  |
+--------------------+
          |
          v
+----------------------------+
|  DWConv (if stride == 2)  |
|     or Identity           |
+----------------------------+
          |
          v
+----------------------------+
|   GhostConv (1x1, linear) |
+----------------------------+
          |
          v
+-----------------------------+
|      Shortcut path:        |
|  DWConv + Conv (if s=2)    |
|      or Identity           |
+-----------------------------+
          |
          v
+----------------------+
|   Add + Output       |
+----------------------+
```

```python
class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
```

## Expand and Contract

### Contract
- Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)


### expand
- Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)



## DetectMultiBackend
在推理过程中，DetectMultiBackend 会调用不同推理后端的实现。例如，使用 PyTorch 后端时，它会调用 model(im) 执行前向推理；而在使用 ONNX 或 TensorRT 时，它会调用对应的推理引擎进行推理。
- 该类会load参数中传入的model，即**yolov5s**
```python

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
```
# model/YOLO
## Detect
### 流程

- Detect层作为最后一层，依然会通过卷积为每个cell，计算出预测结果。`(tx,ty,tw,th，objectness,cls_conf)`。其中cls_conf是互斥的，即可以有多个最大值。detect层中并不知道anchor的存在，只有训练时，会借助anchor将输出解码成框，并且通过反向传播更新参数。推理时也会计算出解码后的框回传。
- train时，只有特征图的输出x.`x:(bs, na, gridy,gridx,  no)`,并且x**未解码**，表示预测的`(tx,ty,tw,th，objectness,cls_conf)`没有与anchor和cell的坐标作用。

- 当推理模式时，输出形状是：`cat((batch_size, na * nx * ny, no),x)`  z是**解码**过后的预测框。(使用sigmoid函数解码)
```text
预测值(每个anchor): [tx, ty, tw, th]
grid:   cell 的中心点坐标
anchor_grid: 每个 cell 的 anchor box 的 w 和 h（相对原图）

最终预测框（x, y, w, h）为：
  x = sigmoid(tx) + grid_x
  y = sigmoid(ty) + grid_y
  w = exp(tw) × anchor_w
  h = exp(th) × anchor_h
```
- height*width：表示特征图的面积，而na：表示每层anchor的个数一般为三个；no：表示anchor预测的数量；
    - 每个anchor都会预测出nc+5个数值[x, y, w, h, objectness] + [cls1, cls2, ..., cls_nc]；我们会找出**匹配度最高(IOU)的anchor用做ground truth**，继续训练model逼近这个预测框；其余都被认为负样本。如果该 anchor 与某个真实框匹配（IoU较高），**它的objectness 标签为1**；
eg:
```perl
(batch_size, na * nx * ny, no)
= (1, 3 * 2 * 2, 8)
= (1, 12, 8)
```
z的具体输出如下：
```python
# 假设预测结果 z 的内容
z = [
    # 图像 1（batch_size=1）
    [
        # 锚框 1 (对于特征图位置 (0,0))
        [x1, y1, w1, h1, conf1, cls1_1, cls1_2, cls1_3],
        # 锚框 2 (对于特征图位置 (0,0))
        [x2, y2, w2, h2, conf2, cls2_1, cls2_2, cls2_3],
        # 锚框 3 (对于特征图位置 (0,0))
        [x3, y3, w3, h3, conf3, cls3_1, cls3_2, cls3_3],
        
        # 锚框 4 (对于特征图位置 (0,1))
        [x4, y4, w4, h4, conf4, cls4_1, cls4_2, cls4_3],
        # 锚框 5 (对于特征图位置 (0,1))
        [x5, y5, w5, h5, conf5, cls5_1, cls5_2, cls5_3],
        # 锚框 6 (对于特征图位置 (0,1))
        [x6, y6, w6, h6, conf6, cls6_1, cls6_2, cls6_3],

        # 锚框 7 (对于特征图位置 (1,0))
        [x7, y7, w7, h7, conf7, cls7_1, cls7_2, cls7_3],
        # 锚框 8 (对于特征图位置 (1,0))
        [x8, y8, w8, h8, conf8, cls8_1, cls8_2, cls8_3],
        # 锚框 9 (对于特征图位置 (1,0))
        [x9, y9, w9, h9, conf9, cls9_1, cls9_2, cls9_3],

        # 锚框 10 (对于特征图位置 (1,1))
        [x10, y10, w10, h10, conf10, cls10_1, cls10_2, cls10_3],
        # 锚框 11 (对于特征图位置 (1,1))
        [x11, y11, w11, h11, conf11, cls11_1, cls11_2, cls11_3],
        # 锚框 12 (对于特征图位置 (1,1))
        [x12, y12, w12, h12, conf12, cls12_1, cls12_2, cls12_3]
    ]
]

```
```python
 def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)

                     return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

```


**anchor**
 假设我们输入一张 `640×640` 的图片，经过卷积 backbone 和 FPN 后，最后会得到不同尺度的特征图，例如：
|层级（head）|特征图尺寸|每个 cell 对应原图的大小|stride|
|---|---|---|---|
|P3 (small)|80×80|640/80 = 8 → 8×8 像素|8|
|P4 (medium)|40×40|640/40 = 16 → 16×16 像素|16|
|P5 (large)|20×20|640/20 = 32 → 32×32 像素|32|
- 这张特征图上的每一个位置 `[i, j]`，我们就叫做一个 **cell**；
- 每个 cell 实际代表了原图中一个 `stride × stride` 的区域。所以stride越大，每个点代表的区域也就越大，视野也就越大，适合检测大目标； 每一个cell都配备了**一组anchor框(3个)**，有些anchor很大，所以cell中的grid都会交叉。
每层都有 3 个 anchor，因此：
- P3: 3 × 80 × 80 = **19200** anchor
- P4: 3 × 40 × 40 = **4800**
- P5: 3 × 20 × 20 = **1200**
```python
    anchors:
    # 特征图尺寸(w,h), 注释指的是下采样个数，40 / 8 = 80 × 80
    - [10,13, 16,30, 33,23]  # P3/8
    - [30,61, 62,45, 59,119]  # P4/16
    - [116,90, 156,198, 373,326]  # P5/32
```

- anchor_grid和grid 都是归一化后的，即除以了stride，直到预测时再乘回来。其中`grid`代表了中序的坐标，而`anchor_grid`代表了宽和高。

```python
 def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        # i：当前层索引（例如 P3 是 0，P4 是 1，P5 是 2
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape,2表示中心点坐标
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
```


## BaseModel
**forward_once**
- 看当前的输入是否来自上一层，`m.f ==-1`表示来自上一层；如果不是来自上一层，`m.f`代表了第几层(也可能是列表)
- y是用来存储中间层的结果的，也就是x。
- 剩下的就是正常调用model，指的是把所有层都forward一遍。当model写成modellist就可以用for循环循环一遍.
```python
 for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                x = m(x)  # run
```
```python
 self.model = nn.ModuleList([
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        ])
```
```text
输入 x
   ↓
for m in self.model:
    x = 选取输入 (从y或上一层)
    if profile: 分析 FLOPs + 时间
    x = m(x)  # 正式执行
    if m.i in save: y.append(x)
    if visualize: 画图
   ↓
输出 x（最后一层）

```



## DetectionModel
DetectionModel 是**YOLOv5** 的完整结构，是你训练/推理用的模型类,训练的是它、保存的是它、导出 ONNX/torchscript 也是它。它会解析yaml文件，得出整个YOLO的模型结构**Backbone + Neck + Detect**
`model[-1]`：表示最后层，-1的话从末尾开始计数。
- 有两种forward()函数，一种普通版本的.
**加强forward**
一种对特征图放缩加旋转之后再处理的图。处理完的anchor框有很多重复的，还需要裁剪。
```python
   def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales，方法缩小处理
        f = [None, 3, None]  # flips (2-ud, 3-lr)，翻转处理
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train
```

**初始化偏置**
- 不是这是真看不懂了，假设原始图像平均有8个目标；以及每个类的概率大约为$\frac{0.6}{nc}$
```python
  def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

```
## parse_model
- 这个确实可以先不学，后面再学即可。
**输入**:
`def parse_model(d, ch)`d：是 model.yaml 文件解析出来的字典（dict），包含了 backbone, head, anchors, depth_multiple, width_multiple, nc 等;
ch：是每个模块的输入通道列表（通常从 [3] 开始，即 RGB 三通道）。

**输出**
`return nn.Sequential(*layers), sorted(save)`返回一个 nn.Sequential 模型和需要保存特征图的层索引列表（通常用于后续检测头使用）。

```text
           model.yaml
                ↓
         parse_model(d, ch)
                ↓
  遍历 backbone 和 head 模块定义
                ↓
    每层构造 m_(module)，确定 c2（输出通道）
                ↓
       动态创建 PyTorch 模块（m_）
                ↓
   记录 from、index、参数数、输出通道数
                ↓
          构造 nn.Sequential 模型
                ↓
    → 返回 模型结构 + 中间层保存索引（save）

```



### YOLOv5s
**backbone:**
- 从输入图像中提取特征
**head:**
- 融合特征图，并准备目标检测的不同尺度输
- 它就像“视觉逻辑推理中心”，把从不同位置学到的图像知识拼接起来，形成对目标的“理解”。融合特征图，并准备目标检测的不同尺度输

假设你要检测图中的猫：

- Backbone 提取了“猫耳朵”“猫毛”“轮廓”等图像低级特征；
- Head 把不同层的特征拼在一起，理解这是“猫的整体”；
- Detect 输出预测框：“这里有只猫，概率 0.92，框的位置是 (x, y, w, h)”。




# Train
Train的目的为使model**更新各种参数**，学会提取需要识别的图像的**特征**，最终能独立判断出图像中需要框出来的部分。
以下为Train的**具体流程**：
- 首先，Train前需要load数据和model，以及定义好loss function, optimizer, scheduler；还需要warm up。dataloader会得到`(imgs, targets, paths, _)`,targets为`[image_index, class, x_center, y_center, width, height]（全是归一化坐标）`，即我们使用的labels。
- 其次,开始时model会随机预测出每个cell中的prediton(offset)，prediction的大小为`[batch_size, na, grid_x,grid_y, no ]`
- 接着，我们计算loss，选择与目标框接近的anchor为基点，这样model学习的难度较小(可能多个anchor对一个target)。正样本 objectness的学习目标为pbox和tbox的iou(因为相交面积代表了我们对这里是否有目标的信心)，负样本objectness的学习目标为0；box的学习目标为tbox(目标box相对于选中cell的offset)；class的学习目标为target的class(每个prediction只对应一个class，其他class的概率都为0)。
- 再次我们会通过反向传播，传播梯度，更新参数。
- Remark：YOLO中使用的是混合精度训练(AMP)，大部分情况使用FP16，对精度要求高的操作(loss`scaler.scale(loss).backward()`)，保留FP32，这样可以节约计算资源.为什么不会出现梯度消失呢？计算loss时，我们使用float32，既不会梯度消失(loss乘以1024)，也不会梯度爆炸(原本是FP16，现在是FP32).并且更新权重时，梯度会除以1024，且会限制梯度大小，防止梯度爆炸。
- Remark:YOLO 使用指数滑动平均（EMA）维护一份模型的平滑版本。随着训练迭代次数增加，EMA 对当前模型参数的更新幅度逐渐减小，更依赖过去累积的历史信息，从而使模型在评估时更加稳定，避免因瞬时波动导致性能不稳定。
- 最后，我们会使用val验证model的功效，具体对不同的val的结果使用不同的权重，并且使用`fitness`定量。我们会记录下最高的fitness对应的model参数，即`best.pt`

```text
Train epoch
↓
Update LR scheduler
↓
Update EMA model 属性
↓
验证集评估 (validate)
↓
更新 best mAP（fitness）
↓
Early stopping 检查
↓
Callbacks 日志记录

```

## Loss
`class Computeloss`
用于计算损失,首先我们肯定不用所有的prediction计算损失，我们先挑选**适合学习**的anchor和cell。
- `build_targets(self, p, targets)`:找出比较适合用于学习(尺寸筛选较适合)`j = torch.max(r, 1 / r).max(2)[0] < anchor_t`的anchor框的大小和cell的idx。(p的作用是提供cell的大小)，`return tcls, tbox, indices, anch`.
Remark:我们是针对于图中的每个label选择的anchor和对应的cell，所以可能出现**多个anchors对应一个label**。
    - tcls:为class，一般层数为3，包含该层所有正样本的类别id，从target class中选择出来。类别根据target中的label选择出来，**每个anchor需要学习box和对一个的label**
    - tbox：为目标框在当前cell的offset与宽高`[x_offset, y_offset, width, height]`
    - indices：`[b,a,gj,gi]`表示imgidx,anchoridx,gridxidx, gridyidx.
    - anch：表示anchor的大小`[width,height]`。
```python
def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
 # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t'] 
 # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

```
- 一共有三种损失，**class，box(位置损失)**,**objectness(目标存在损失)**,只有被build_targets选中的prediction`  pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  `会计算class和box损失。
- box loss采取`pbox,tbox`的iou loss的平均值`lbox += (1.0 - iou).mean`
```python
 pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
```
- objectness loss 采取**交叉熵**，**没有被选中的predction的objectness为0**。没有被选中的anchor也需要学习目标是否存在这件事情。并且我们希望选择的predction最后得出的conf和`pbox,tbox`的iou一致
```python
 # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio
obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
```
- label smoothing: 标签平滑，将0和1smoothing，提高模型的泛化能力，并且使model学习得不会过于绝对。
- class loss: 预测出的loss与目标loss的二分类交叉熵(只对正样本进行)，并且是针对索引进行的,我们希望训练出来只有`tcls[i]`中的类别被预测出来。`tcls[i]`表示的是第i层中，选中的anchor对应的真实类别(共有n种)，且我们选取的pcls每层也只有n种。
remark：只有一类的class就不需要这个loss了；且loss里的循环针对的是anchor的层数(3层)。
```python
 for i, pi in enumerate(p): 
    # layer index, layer predictions
       # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE
```

- `pi[b, a, gj, gi]`:因为x输出的形状为`[batch,na,grid_y,grid_x,no]`
```python
pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)

# 计算的是总损失*bs
# 后一个输出把他们拼接成一个张量是为了输出，所有使用.detach()，不迭代梯度。 
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
```








## EMA
- (指数加权移动平均)是一种常用的技术，用于平滑和稳定模型训练过程中的参数更新,普遍用于推理model。
- 首先，它与原模型共享结构，但参数是独立的副本，不会参与梯度计算(关闭EMA的参数的梯度更新).
```python
 def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
```
- 基本的更新公式为：$\hat{θ}_t​=d⋅\hat{θ}_(t−1)​+(1−d)⋅\hat{θ}_t​$。随着update的增加，衰减因子d不断增大，d越大，过去EMA的计算占比越大，当前的主模型参数占比就越小。 `v` 是 EMA 模型中的某一个参数张量,主模型参数就是 `msd[k]`。让模型前期快速融合，后期逐
```python
number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        self.updates += 1
        d = self.decay(self.updates)

        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
```

```python
 
    def update(self, model):
        # Update EMA(Exponential Moving Average)parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict，获取所有参数的权重。
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
```


## AMP
- 混合精度训练，混合精度指的是混合float16和float32精度。为了节省内存和加速，采用float16，但float16的精度太低，为了防止梯度underflow，需要临时把loss放大。再之后，再把梯度缩回来，就可以正常更新了。
- backward()用于计算梯度;`scaler.step(optimizer)`用于更新梯度
- 需要配合`autocast`使用，autocast 是 PyTorch 中用于 自动混合精度训练 的上下文管理器(自动决定哪些计算使用float16和float32)
> - 前向传播 16
> - loss 32，要求精度高
> - 反向传播 16，但是要乘以一个scaler
- 放大为什么不会导致梯度爆炸：因为我们传播时虽然扩大了，但是我们更新参数时缩小回来了，还将最大值裁剪至10，故不会梯度爆炸。
```python
 with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
```

## checkpoint
```python
ckpt = {
    'epoch': epoch,                         # 当前训练到了第几个 epoch
    'best_fitness': best_fitness,           # 当前为止的最佳模型性能（通常基于 mAP）
    'model': deepcopy(de_parallel(model)).half(),  # 当前模型的权重（注意是半精度 float16）
    'ema': deepcopy(ema.ema).half(),        # EMA 平滑后的模型权重（也是 float16）
    'updates': ema.updates,                 # EMA 的更新次数（关键）
    'optimizer': optimizer.state_dict(),    # 当前优化器的状态（包括动量、学习率等）
    'opt': vars(opt),                       # 当前训练用的参数设置（命令行参数）
    'git': GIT_INFO,                        # 当前代码仓库的 Git 信息（可选）
    'date': datetime.now().isoformat()      # 保存时间戳
}

```

## Evaluation
- 先通过valuation计算出p，R，MAP等，再通过`fitness`函数计算出具体的得分.比较当前 fitness 和历史最高 fitness：

如果当前 fitness 高于之前的最高值，就更新保存“最佳模型权重”（best model weights）。否则不保存，继续训练。
```python
torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
```
```python
# Update best mAP
def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
```



## save model
```python
# Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}
```

##  Multi-scale
- YOLOv5会在训练时动态调整图片尺寸，提高model的robust(也可以适用于其他model)
```python
# Multi-scale
if opt.multi_scale:
    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
    sf = sz / max(imgs.shape[2:])  # scale factor
    if sf != 1:
        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

```
## Optimizer
- 前面的代码在处理小batch的情况，防止batch过小导致精度过低。
```python
 # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
```
```python
def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')
```


## scheduler
[Pytorch scheduler] (https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html)
- 余弦退火或者线性退火函数。
- scheduler只改变学习率这一个参数
- `lrf`:learning rate final
```python
# Scheduler
if opt.cos_lr:
    lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
else:
    lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
```

```python
 scheduler.step()
```

# detect
- 每张图片会经过一次detect，找出存在的目标
**warm up**
- 通过一次推理进行预热的主要目的是为了**优化硬件和软件的初始化**，确保模型在实际推理时达到最佳性能。

## 非极大值抑制
NMS 通过以下步骤来完成这个任务：
- 首先对每个框选择概率最大的类，且$conf = obj_conf * cls_conf$,`box = xywh2xyxy(x[:, :4]) `讲坐标具体转换为xy坐标系
- 其次合成x:`[位置，置信度，类别]`总共6个参数。并且置信度要大于thereshold.
- 关键点：调用**Pytorch**中的**NMS**` i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS`，保留得分最高的框，且删除与它iou重合度高于threshold的其余框
- 输出即为output，就是上一步选择出来的优选框`output[xi] = x[i]`
- 如果合并框的话，会计算每个优选框与所有框的iou，并于conf相乘，作为权重。
```python
def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms() # 每张图片anchors的最大值
    for xi, x in enumerate(prediction):  # image index, image ,对batch_size做循环，每个batch_size都有(na*nx*ny,no)的输出
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
    else:  # best class only，对第1维(类别维度)求最大值
            conf, j = x[:, 5:mi].max(1, keepdim=True)# j为idx(即具体是哪个类别)，conf为置信度
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres] # 合成[位置，置信度，类别]且只保留置信度大于threshold的框
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
```
`x[..., 0] `表示取第0列，当矩阵维数很高(5维)时比较方便


# valuation
valuation的输入是model和图片以及对应的label，输出是模型预测的各种精度。
- 首先需要保证输入的数据也要按xywh；其次和inference顺序一样，先调用model，计算出预测值`[tx,ty,tw,th,conf,label]`,若是训练还需要计算loss。接着使用NMS，计算出最大值抑制后的框。
- 现在是val的关键函数`process_batch`,计算label和detections，并且使label和detections一一对应；最后回传不同的IOU阈值下，正确分类的label。`return torch.tensor(correct, dtype=torch.bool, device=iouv.device)`， `correct[i][j]`，表示第i个预测在Iou[j]阈值下是否为TP
- 最后计算各种精度的结果[精度计算](#精度计算),得到最终的结果
`training = model is not None`：特殊的与非，就和 not true类似。
```md
🔁 匹配过程演示：
假设：

有 2 个 GT（label0 和 label1）

模型预测出了 3 个框（pred0, pred1, pred2）

所有预测框与标签都匹配类别，IoU 如下：

label0	label1
pred0	0.85	0.20
pred1	0.60	0.70
pred2	0.88	0.90

在 IoU 阈值为 0.5 下，满足 IoU 和类别匹配条件的候选匹配有：

pred0 ↔ label0

pred1 ↔ label0

pred1 ↔ label1

pred2 ↔ label0

pred2 ↔ label1

🧹 第一次去重：每个 detection 只能对应一个 label
按 IoU 排序，保留每个 detection 匹配中 IoU 最大的：

pred0 ↔ label0（0.85）

pred1 ↔ label1（0.70）

pred2 ↔ label1（0.90）

🧹 第二次去重：每个 label 只能对应一个 detection
label1 现在被 pred1 和 pred2 同时“匹配”，我们选 pred2（IoU 高）

最终匹配对：

pred0 ↔ label0

pred2 ↔ label1

结果中：

pred1 没有成功匹配（虽然有机会），被视为 False Positive

两个 label 都被命中


```
```python
iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
 confusion_matrix = ConfusionMatrix(nc=nc)
    dt = Profile(), Profile(), Profile()  # profiling times
    with dt[0]:
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

     if single_cls:
                pred[:, 5] = 0# 如果只有一个类别的话，全部都算一类

with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)# val 也要NMS


  # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions(after nms)
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1


  # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)


```

```python
def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)
```
# Results
## ConfusionMatrix
混淆矩阵记录预测类别vs真实类别的数量。`matrix[pred_class][true_class]`
- matchs是一个`(k,3)`的数组，列分别为`[ground-truth idx, decetion idx, IOU]`表示预测框idx`m1`预测了实框的idx`m0`，和具体的IOU值。
- 从gronud-truth中循环，找预测出的真实框与原始真实框是否有相同电，有相同认为这是一个TP，否则把它当成背景板。
```python
   m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background
```
```python
def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background
```


## 精度计算
- 必须得会啊，不然你怎么知道哪些精度具体代表了什么呢？amazing， 竟然是这么计算的啊。
- `Recall=TP/nl`:所有正确的类别中预测正确类别的比例,不想漏检(宁滥勿缺)。
- `Precision = TP/TP+NP`：所有预测的类别中预测正确的类别(可能将负样本预测为正样本)，不想错抓(宁缺勿滥)。
- PR曲线：横坐标为Recall,纵坐标为Precision，越靠左上角越好。因为我们现在预测都是根据概率进行，最理想情况为(1,1),但随着Recall的增加，逐渐把所有正确的类别都预测出来，此时必须要牺牲一些精度，把一些很可能负样本预测为正样本(例如只有0.2的概率是绿灯，我们也算)，以提高预测出的正确类别的总量。
    - 若需要提高精度，我们也必须牺牲Recall,只要有一点可能被预测为负样本的正样本我们都不要(就算有0.8概率是绿灯我们也不要)。
```md
举个例子：
你在检测“猫”，模型输出很多候选框，每个框有个置信度（confidence）：

如果你 提高置信度阈值（比如只保留 >0.9 的框）：

✔️ 你保留的预测大多数都是真猫（Precision 高了）

❌ 但很多真正的猫被你过滤掉了（Recall 低了）

如果你 降低置信度阈值（比如只要 >0.2 就算猫）：

✔️ 你几乎不会漏猫了（Recall 高了）

❌ 但你预测的一堆“猫”中有很多是假猫（Precision 降低）
```
- `AP(Average Precision)`:PR曲线下的面积，表示平均精度。(YOLO检测时会有不同IOU阈值对应的精度)
- `MAP(mean Average Precision)`:所有类别的平均精度(MAP@0.5)表示IOU阈值为0.5时的MPA，`MAP@[.5:.95]`表示阈值from 0.5 to 0.95的MAP的平均值。
| 类别  | AP@0.5 | AP@0.55 | ... | AP@0.95 |
| --- | ------ | ------- | --- | ------- |
| cat | 0.80   | 0.75    | ... | 0.40    |
| dog | 0.78   | 0.72    | ... | 0.35    |
| car | 0.70   | 0.68    | ... | 0.30    |
MAP@0.5= (0.80+0.78+0.70)/3 = 0.76
- `F1 = (2*P*R)/(P+R+eps)`：Precision 和 Recall的调和平均，为了平衡Precision和Recall，调和平均是因为它可以惩罚两者不平衡的情况，若一方较低，则F1较低，只有在两者都高分时，F1才高。
| Precision | Recall | 算术平均 | 调和平均（F1） |
| --------- | ------ | ---- | -------- |
| 1.0       | 0.0    | 0.5  | 0.0 ❌    |
| 0.9       | 0.9    | 0.9  | 0.9 ✅    |
| 0.6       | 0.3    | 0.45 | 0.4      |
| 0.9       | 0.1    | 0.5  | 0.18 ❌   |

```python
def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=""):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i] # 按索引重新排序

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True) # nt为每个类别真实目标的数量
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)
```

## scale_boxes
YOLO一般会对图像进行resize，统一到`640*640`
```python
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None)
# Rescale boxes (xyxy) from img1_shape to img0_shape
```

## box_iou
```python
def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
```
