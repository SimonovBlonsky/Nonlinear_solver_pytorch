import torch
from collections import defaultdict

class ImageFrame:
    def __init__(self, points=None, t=0.0):
        # points 是一个字典，key 为 int 类型，value 为一个包含 (int, 7x1 向量) 的列表
        self.points = points if points is not None else defaultdict(list)
        
        # 时间戳 t
        self.t = t
        
        # 旋转矩阵 R 使用 torch.Tensor
        self.R = torch.eye(3)  # 3x3 单位矩阵
        
        # 平移向量 T 使用 torch.Tensor
        self.T = torch.zeros(3)  # 3x1 零向量
        
        # 预积分对象
        self.pre_integration = None
        
        # 是否为关键帧
        self.is_key_frame = False