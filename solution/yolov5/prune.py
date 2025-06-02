import torch
from utils.torch_utils import prune  # 确保你的 YOLO 项目里有这个函数

# ==== 1. 加载模型 ====
model = torch.load('your_model_path.pt', map_location='cuda')  # 替换为你的模型路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ==== 2. 模型剪枝 ====
prune(model, amount=0.3)  # 剪除 30% 的不重要权重（按绝对值）

# ==== 3. 切换评估模式 ====
model.eval()

# ==== 4. 数据配置 ====
# 假设你有 data.yaml 对应字典结构（加载或写死均可）
data = {
    'val': 'data/coco/val2017.txt',  # 验证集路径
    'nc': 80                         # 类别数（根据你的数据集修改）
}
single_cls = False  # 如果只检测一个类别就设为 True

# ==== 5. 判断数据集类型 ====
is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')

# ==== 6. 类别数设置 ====
nc = 1 if single_cls else int(data['nc'])  # 用于检测头和评估指标

# ==== 7. mAP 评估相关的 IoU 阈值 ====
iouv = torch.linspace(0.5, 0.95, 10).to(device)  # [0.5, 0.55, ..., 0.95]
niou = iouv.numel()  # 一共 10 个阈值

# ==== 8. 打印信息确认 ====
print(f'剪枝完成: 保留 {(1 - 0.3) * 100:.1f}% 权重')
print(f'使用{"COCO" if is_coco else "自定义"}数据集，类别数: {nc}')
print(f'使用 IoU 阈值向量: {iouv.tolist()} (用于 mAP@0.5:0.95)')
