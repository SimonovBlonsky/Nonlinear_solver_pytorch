import torch
import numpy
import Utils
from enum import Enum

from BaseFactor import NonlinearFactor
from dpvo.viba.IntegrationBase import IntegrationBase
from dpvo.lietorch import SO3

G = 9.807

O_P = 0
O_R = 3
O_V = 6
O_BA = 9
O_BG = 12
I_3x3 = torch.eye(3, dtype=torch.float, device="cuda")

class IMUFactor(NonlinearFactor):
    def __init__(self, pre_integration):
        super().__init__()
        
        # Preintegration
        self.pre_integration_ = pre_integration
        
        # Jacobians
        self.dp_dba_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
        self.dp_dbg_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
        self.dr_dbg_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
        self.dv_dba_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
        self.dv_dbg_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
    
    def ComputeResidual(self):
        
        # Vins-course uses, sophus, we use torch here
        # Pose is represented by (P, q)
        # q denotes quaternion (w, x, y, z)
        
        # Param 0 is related to pose i
        param_0 = self.vertices_[0].parameters()
        Qi = torch.tensor([param_0[6], param_0[3], param_0[4], param_0[5]], dtype=torch.float, device="cuda")
        Qi = SO3(Qi)
        Pi = torch.tensor([param_0[0], param_0[1], param_0[2]], dtype=torch.float, device="cuda")

        # Param 1 is related to v,b i
        param_1 = self.vertices_[1].parameters()
        Vi = torch.tensor([param_1[0], param_1[1], param_1[2]], dtype=torch.float, device="cuda")
        Bai = torch.tensor([param_1[3], param_1[4], param_1[5]], dtype=torch.float, device="cuda")
        Bgi = torch.tensor([param_1[6], param_1[7], param_1[8]], dtype=torch.float, device="cuda")
        
        # Param 2 is related to pose j
        param_2 = self.vertices_[2].parameters()
        Qj = torch.tensor([param_2[6], param_2[3], param_2[4], param_2[5]], dtype=torch.float, device="cuda")
        Qj = SO3(Qj)
        Pj = torch.tensor([param_2[0], param_2[1], param_2[2]], dtype=torch.float, device="cuda")

        # Param 3 is related to v, b j
        param_3 = self.vertices_[3].parameters()
        Vj = torch.tensor([param_3[0], param_3[1], param_3[2]], dtype=torch.float, device="cuda")
        Baj = torch.tensor([param_3[3], param_3[4], param_3[5]], dtype=torch.float, device="cuda")
        Bgj = torch.tensor([param_3[6], param_3[7], param_3[8]], dtype=torch.float, device="cuda")
        
        # get residual
        self.residual_ = self.pre_integration_.evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj)
        
        # get information matrices w.r.t state variables and measurement noise
        self.setInformation(self.pre_integration_.covariance.inv())
        
    def ComputeJacobians(self):
        param_0 = self.vertices_[0].parameters()
        Qi = torch.tensor([param_0[6], param_0[3], param_0[4], param_0[5]], dtype=torch.float, device="cuda")
        Qi = SO3(Qi)
        Pi = torch.tensor([param_0[0], param_0[1], param_0[2]], dtype=torch.float, device="cuda")

        param_1 = self.vertices_[1].parameters()
        Vi = torch.tensor([param_1[0], param_1[1], param_1[2]], dtype=torch.float, device="cuda")
        Bai = torch.tensor([param_1[3], param_1[4], param_1[5]], dtype=torch.float, device="cuda")
        Bgi = torch.tensor([param_1[6], param_1[7], param_1[8]], dtype=torch.float, device="cuda")
        
        param_2 = self.vertices_[2].parameters()
        Qj = torch.tensor([param_2[6], param_2[3], param_2[4], param_2[5]], dtype=torch.float, device="cuda")
        Qj = SO3(Qj)
        Pj = torch.tensor([param_2[0], param_2[1], param_2[2]], dtype=torch.float, device="cuda")

        param_3 = self.vertices_[3].parameters()
        Vj = torch.tensor([param_3[0], param_3[1], param_3[2]], dtype=torch.float, device="cuda")
        Baj = torch.tensor([param_3[3], param_3[4], param_3[5]], dtype=torch.float, device="cuda")
        Bgj = torch.tensor([param_3[6], param_3[7], param_3[8]], dtype=torch.float, device="cuda")
        
        sum_dt = self.pre_integration_.sum_dt
        
        # Extract jacobians from pre integration
        dp_dba = self.pre_integration_.jacobian[O_P:O_P+3, O_BA:O_BA+3]
        dp_dbg = self.pre_integration_.jacobian[O_P:O_P+3, O_BG:O_BG+3]
        dq_dbg = self.pre_integration_.jacobian[O_R:O_R+3, O_BG:O_BG+3]
        dv_dba = self.pre_integration_.jacobian[O_V:O_V+3, O_BA:O_BA+3]
        dv_dbg = self.pre_integration_.jacobian[O_V:O_V+3, O_BG:O_BG+3]
    
        if self.pre_integration_.jacobian.max().item() > 1e8 or self.pre_integration_.jacobian.min().item() < -1e8:
            Warning("numerical unstable in preintegration")
        
        ### The following computes IMU preintegration error jacobian w.r.t pose, speed and bias
        # self.jacobians_[0] w.r.t pose i
        jacobian_pose_i = torch.zeros(15, 6)
        jacobian_pose_i[O_P:O_P+3, O_P:O_P+3] = -Qi.inv().matrix()
        jacobian_pose_i[O_P:O_P+3, O_R:O_R+3] = Utils.skew_symmetric(Qi.inv() @ (0.5 * G * sum_dt**2 + Pj - Pi - Vi * sum_dt))

        corrected_delta_q = self.pre_integration_.delta_q * Utils.deltaQ(dq_dbg * (Bgi - self.pre_integration_.linearized_bg))
        jacobian_pose_i[O_R:O_R+3, O_R:O_R+3] = -(Utils.Qleft(Qj.inv() * Qi) @ Utils.Qright(corrected_delta_q))[-3:, -3:]  # 修正的四元数差异

        jacobian_pose_i[O_V:O_V+3, O_R:O_R+3] = Utils.skewSymmetric(Qi.inv() * (G * sum_dt + Vj - Vi))

        self.jacobians_[0] = jacobian_pose_i
        
        # self.jacobians_[1] w.r.t speedbias i
        jacobian_speedbias_i = torch.zeros(15, 9, dtype=torch.float, device="cuda")
        jacobian_speedbias_i[O_P:O_P+3, (O_V-O_V):(O_V-O_V+3)] = -Qi.inv().matrix() @ sum_dt
        jacobian_speedbias_i[O_P:O_P+3, (O_BA - O_V):(O_BA - O_V + 3)] = -dp_dba
        jacobian_speedbias_i[O_P:O_P+3, (O_BG - O_V):(O_BG - O_V + 3)] = -dp_dbg
        jacobian_speedbias_i[O_R:O_R+3, (O_R - O_V):(O_BG - O_V + 3)] = -Utils.Qleft(Qj.inv() * Qi * self.pre_integration_.delta_q)[-3:, -3:] @ dq_dbg
        
        jacobian_speedbias_i[O_V:O_V+3, (O_V - O_V):(O_V - O_V + 3)] = -Qi.inv().matrix()
        jacobian_speedbias_i[O_V:O_V+3, (O_BA - O_V):(O_BA - O_V + 3)] = -dv_dba
        jacobian_speedbias_i[O_V:O_V+3, (O_BG - O_V):(O_BG - O_V + 3)] = -dv_dbg
        
        jacobian_speedbias_i[O_BA:O_BA+3, (O_BA - O_V):(O_BA - O_V + 3)] = -I_3x3
        jacobian_speedbias_i[O_BG:O_BG+3, (O_BG - O_V):(O_BG - O_V + 3)] = -I_3x3
    
        self.jacobians_[1] = jacobian_speedbias_i
        
        # self.jacobians_[2] w.r.t pose j
        jacobian_pose_j = torch.zeros(15, 6, dtype=torch.float, device="cuda")
        jacobian_pose_i[O_P:O_P+3, O_P:O_P+3] = -Qi.inv().matrix()
        corrected_delta_q = self.pre_integration_.delta_q * Utils.deltaQ(dq_dbg * (Bgi - self.pre_integration_.linearized_bg))
        jacobian_pose_i[O_R:O_R+3, O_R:O_R+3] = Utils.Qleft(corrected_delta_q.inverse() * Qi.inv() * Qj)[-3:,-3:]
        self.jacobians_[2] = jacobian_pose_j
        
        # self.jacobians_[3] w.r.t speedbias j
        jacobian_speedbias_j = torch.zeros(15, 9, dtype=torch.float, device="cuda")
        jacobian_speedbias_j[O_P:O_P+3, (O_V-O_V):(O_V-O_V+3)] = Qi.inv().matrix()
        jacobian_speedbias_j[O_BA:O_BA+3, (O_BA - O_V):(O_BA - O_V + 3)] = I_3x3
        jacobian_speedbias_j[O_BG:O_BG+3, (O_BG - O_V):(O_BG - O_V + 3)] = I_3x3
        self.jacobians_[3] = jacobian_pose_j