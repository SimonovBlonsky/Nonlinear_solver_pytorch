import torch
import LossFunction

from BaseState import Vertex

class NonlinearFactor:
    def __init__(self, residual_dimension, num_States):
        
        self.residual_dimension = residual_dimension
        self.num_States = num_States
        self.id_ = 0
        self.orderingId_ = 0
        
        # States (vertices)
        self.vertices_ = []
        self.vertices_types_ = None
        
        # loss functions
        self.loss_function_ = None
        self.Chi2 = 0.0
        self.robustChi2 = 0.0
        
        # Jacobian and residual
        self.jacobians_ = None
        self.residual_ = None
        
        # info mat
        self.information_ = None
        self.sqrt_information_ = None
        
        self.observations_ = None
        
    
    def Id(self): return self.id_
    
    def addVertex(self, state):
        self.vertices_.append(state)
        return True
    
    def setVertex(self, states):
        self.vertices = states
        return True
    
    def getVertex(self, i): return self.vertices_[i]
    
    def vertices(self): return self.vertices_
    
    def numVertices(self): return len(self.vertices_)
    
    # Type info
    
    # Compute Jacobian
    
    # Compute residual
    
    def Chi2(self): pass
    def RobustChi2(self):pass
    
    def residual(self): return self.residual_
    
    def jacobians(self): return self.jacobians_
    
    def setInformation(self, information):
        self.information_ = information
        try:
            # PyTorch 的 Cholesky 返回下三角矩阵
            L = torch.linalg.cholesky(information)
            self.sqrt_information_ = L.T  # 转置得到上三角矩阵
        except RuntimeError as e:
            raise ValueError(f"Cholesky decomposition failed: {e}")
    
    def information(self): return self.information_
    def sqrt_information(self): return self.sqrt_information_
    
    def setLossFunction(self, ptr): self.loss_function_ = ptr
    def getLossFunction(self): return self.loss_function_
    
    def getObservations(self): return self.observations_
    
    # check vertices
    def checkValid(self): return True
    
    def orderingId(self): return self.orderingId_