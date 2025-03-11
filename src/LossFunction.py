import torch
import math

class TrivalLoss:
    def __init__(self):
        pass
    def compute(self, err2, rho):
        rho[0] = err2
        rho[1] = 1.0
        rho[2] = 0.0
    
class HuberLoss:
    def __init__(self, delta):
        self.delta_ = delta
    def compute(self, e, rho: torch.Tensor):
        dsqr = self.delta_ * self.delta_
        if e <= dsqr:
            rho[0] = e
            rho[1] = 1.0
            rho[2] = 0.0
        else:
            sqrte = math.sqrt(e)
            rho[0] = 2*sqrte*self.delta_ - dsqr
            rho[1] = self.delta_ / sqrte
            rho[2] = -0.5 * rho[1] / e
        
    
class CauchyLoss:
    def __init__(self, delta):
        self.delta_ = delta
    def compute(self, err2, rho: torch.Tensor):
        dsqr = self.delta_ * self.delta_
        dsqrReci = 1.0 / dsqr
        aux = dsqrReci * err2 + 1.0
        rho[0] = dsqr * math.log(aux)
        rho[1] = 1. / aux
        rho[2] = -dsqrReci * math.pow(rho[1], 2)
    
class TukeyLoss:
    def __init__(self, delta):
        self.delta_ = delta
    def compute(self, e2, rho):
        e = math.sqrt(e2)
        delta2 = self.delta_ * self.delta_
        if e <= self.delta_:
            aux = e2 / delta2;
            rho[0] = delta2 * (1. - math.pow((1. - aux), 3)) / 3.
            rho[1] = math.pow((1. - aux), 2)
            rho[2] = -2. * (1. - aux) / delta2
        else:
            rho[0] = delta2 / 3.
            rho[1] = 0
            rho[2] = 0