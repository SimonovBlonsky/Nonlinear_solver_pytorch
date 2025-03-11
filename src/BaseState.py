import torch
import copy
class Vertex:
    def __init__(self):
        self.parameters_ = None
        self.parameters_backup_ = None  # 每次迭代优化中对参数进行备份，用于回滚
        self.local_dimension_ = 1
        
        ### id ###
        self.id_ = 0    # 顶点的id，自动生成
        self.ordering_id_ = 0 # ordering id是在problem中排序后的id，用于寻找雅可比对应块. ordering id带有维度信息，例如ordering_id=6则对应Hessian中的第6列
        
        self.fixed_ = False
        
    def dimension(self): pass
    
    def LocalDimension(self): pass
    
    def id(self): return self.id_
    
    def parameters(self): return self.parameters_
    def setParameters(self, params): self.parameters_ = params
    def backupParameters(self): self.parameters_backup_ = self.parameters_
    def rollbackParameters(self): self.parameters_ = self.parameters_backup_
    
    # plus
    def plus(self): pass
    
    # Type info
    
    # set id
    def orderingId(self): return self.ordering_id_
    
    def setOrderingId(self, id): self.ordering_id_ = id
    
    def setFixed(self, fixed=True):
        self.fixed_ = fixed
    
    def isFixed(self): return self.fixed_
    
    