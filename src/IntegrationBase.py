import torch
import numpy as np
import Utils

G = 9.807

# State order
O_P = 0
O_R = 3
O_V = 6
O_BA = 9
O_BG = 12

# Noise order
O_AN = 0
O_GN = 3
O_AW = 6
O_GW = 9

# Noise params
ACC_N = 0.2
GYR_N = 0.016968
ACC_W = 0.003
GYR_W = 0.000019393
INT_N = 0.316227

class IntegrationBase:
    def __init__(self, acc_0, gyr_0, linearized_ba, linearized_bg, imu_confidence=None):
        
        # raw measurements
        self.dt = 0
        self.acc_0 = acc_0
        self.gyr_0 = gyr_0
        self.acc_1 = 0
        self.acc_1 = 0
        
        # linearized measurements
        self.linearized_acc = acc_0
        self.linearized_gyr = gyr_0
        self.linearized_ba = linearized_ba
        self.linearized_bg = linearized_bg
        
        # Propagation matrices
        self.jacobian = torch.eye(15)
        self.covariance = torch.zeros(15, 15)

        # Noise
        self.noise = torch.zeros(18, 18)
        self.noise[0:3, 0:3] = (ACC_N * ACC_N) * torch.eye(3)
        self.noise[3:6, 3:6] = (GYR_N * GYR_N) * torch.eye(3)
        self.noise[6:9, 6:9] = (ACC_N * ACC_N) * torch.eye(3)
        self.noise[9:12, 9:12] = (GYR_N * GYR_N) * torch.eye(3)
        self.noise[12:15, 12:15] = (ACC_W * ACC_W) * torch.eye(3)
        self.noise[15:18, 15:18] = (GYR_W  * GYR_W ) * torch.eye(3)
        
        # Confidence
        self.confidence = imu_confidence
        
        # Preintegration
        # Here use rotation matrix to represent q
        self.sum_dt = 0.0
        self.delta_p = torch.zeros(3)
        self.delta_q = torch.eye(3)
        self.delta_v = torch.zeros(3)

        # measurement buffer
        self.dt_buf = []
        self.acc_buf = []
        self.gyr_buf = []
        
        # For checking jacobians
        self.step_jacobian = torch.eye(15)
        self.step_V = torch.zeros(15, 18)

    # Add measurements and propagate integration
    def push_back(self, dt, acc, gyr):
        self.dt_buf.append(dt)
        self.acc_buf.append(acc)
        self.gyr_buf.append(gyr)
        self.propagate(dt, acc, gyr)

    # When bias updated, re-do the propagation
    def repropagate(self, _linearized_ba, _linearized_bg):
        self.sum_dt = 0.0
        self.acc_0 = self.linearized_acc
        self.gyr_0 = self.linearized_gyr
        self.delta_p = torch.zeros(3)
        self.delta_q = torch.eye(3)  # Set identity
        self.delta_v = torch.zeros(3)
        self.linearized_ba = _linearized_ba
        self.linearized_bg = _linearized_bg
        self.jacobian = torch.eye(15)   # Set identity
        self.covariance = torch.zeros(15, 15)   # Set zero

        for i in range(len(self.dt_buf)):
            self.propagate(self.dt_buf[i], self.acc_buf[i], self.gyr_buf[i])

    # Midpoint integration logic
    def midPointIntegration(self, _dt, _acc_0, _gyr_0, _acc_1, _gyr_1, \
                delta_p, delta_q, delta_v, linearized_ba, linearized_bg, update_jacobian=False):
        
        print("midpoint integration")
        # Compute mid-point acceleration, angular-velocity
        un_acc_0 = delta_q @ (_acc_0 - linearized_ba)
        un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg
        
        # TODO: test quaternion
        d_delta_q = Utils.q2R(torch.tensor([1, un_gyr[0] * _dt/2, un_gyr[0] * _dt/2 , un_gyr[2] * _dt/2]))
        result_delta_q = delta_q @ d_delta_q
            
        un_acc_1 = result_delta_q @ (_acc_1 - linearized_ba)
        un_acc = 0.5 * (un_acc_0 + un_acc_1)

        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt
        result_delta_v = delta_v + un_acc * _dt

        result_linearized_ba = linearized_ba
        result_linearized_bg = linearized_bg
        
        # update jacobian
        if(update_jacobian):
            print("update jacobian")
            w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg
            a_0_x = _acc_0 - linearized_ba
            a_1_x = _acc_1 - linearized_ba

            # hat
            R_w_x = torch.tensor(
                [[0, w_x[0], -w_x[1]],
                 [-w_x[0], 0, w_x[2]],
                 [w_x[1], -w_x[2], 0]]
                )
            
            R_a_0_x = torch.tensor(
                [[0, a_0_x[0], -a_0_x[1]],
                 [-a_0_x[0], 0, a_0_x[2]],
                 [a_0_x[1], -a_0_x[2], 0]]
                )
            
            R_a_1_x = torch.tensor(
                [[0, a_1_x[0], -a_1_x[1]],
                 [-a_1_x[0], 0, a_1_x[2]],
                 [a_1_x[1], -a_1_x[2], 0]]
            )
            
            # F and V
            F = torch.zeros(15, 15)
            V = torch.zeros(15, 18)

            # Update F w.r.t states
            # delta_q belows is delta_q.toRotationMatrix
            F[0:3, 0:3] = torch.eye(3)
            F[0:3, 3:6] = -0.25 * delta_q @ R_a_0_x * _dt**2 + \
                        -0.25 * result_delta_q @ R_a_1_x @ (torch.eye(3) - R_w_x * _dt) * _dt**2
            F[0:3, 6:9] = torch.eye(3) * _dt
            F[0:3, 9:12] = -0.25 * (delta_q + result_delta_q) * _dt**2
            F[0:3, 12:15] = -0.25 * result_delta_q @ R_a_1_x * _dt * _dt * -_dt

            F[3:6, 3:6] = torch.eye(3) - R_w_x * _dt
            F[3:6, 12:15] = -torch.eye(3) * _dt

            F[6:9, 3:6] = -0.5 * delta_q @ R_a_0_x * _dt + \
                        -0.5 * result_delta_q @ R_a_1_x @ (torch.eye(3) - R_w_x * _dt) * _dt
            F[6:9, 6:9] = torch.eye(3)
            F[6:9, 9:12] = -0.5 * (delta_q + result_delta_q) * _dt
            F[6:9, 12:15] = -0.5 * result_delta_q @ R_a_1_x * _dt * -_dt

            F[9:12, 9:12] = torch.eye(3)
            F[12:15, 12:15] = torch.eye(3)

            # Update V w.r.t measurement noise
            V[:3, :3] = 0.25 * delta_q * _dt**2
            V[:3, 3:6] = 0.25 * -result_delta_q @ R_a_1_x * _dt**2 * 0.5 * _dt
            V[:3, 6:9] = 0.25 * result_delta_q * _dt**2
            V[:3, 9:12] = V[:3, 3:6]

            V[3:6, 3:6] = 0.5 * torch.eye(3) * _dt
            V[3:6, 9:12] = 0.5 * torch.eye(3) * _dt

            V[6:9, :3] = 0.5 * delta_q * _dt
            V[6:9, 3:6] = 0.5 * -result_delta_q @ R_a_1_x * _dt * 0.5 * _dt
            V[6:9, 6:9] = 0.5 * result_delta_q * _dt
            V[6:9, 9:12] = V[6:9, 3:6]

            V[9:12, 12:15] = torch.eye(3) * _dt
            V[12:15, 15:] = torch.eye(3) * _dt

            # Update jacobians w.r.t system states
            self.jacobian = F @ self.jacobian
            
            # Update covariance w.r.t system states and measurement noise
            self.covariance = F @ self.covariance @ F.T + V @ self.noise @ V.T

        return result_delta_p, result_delta_q, result_delta_v, result_linearized_ba, result_linearized_bg

    def propagate(self, _dt, _acc_1, _gyr_1):
        self.dt = _dt
        self.acc_1 = _acc_1
        self.gyr_1 = _gyr_1
        result_delta_p, result_delta_q, result_delta_v, \
            result_linearized_ba, result_linearized_bg = \
            self.midPointIntegration(_dt, self.acc_0, self.gyr_0, _acc_1, _gyr_1, \
                                    self.delta_p, self.delta_q, self.delta_v,
                                    self.linearized_ba, self.linearized_bg, True)

        # checkJacobian
        
        # Update system variables
        self.delta_p = result_delta_p
        self.delta_q = result_delta_q
        self.delta_v = result_delta_v
        self.linearized_ba = result_linearized_ba
        self.linearized_bg = result_linearized_bg
        self.delta_q = Utils.NormalizeR(self.delta_q)
        self.sum_dt += self.dt
        
        # updated measurements set to the current measurement
        self.acc_0 = self.acc_1
        self.gyr_0 = self.gyr_1
    
    # Synonym of propagate
    def integrateMeasurement(self, _dt, _acc_1, _gyr_1):
        self.propagate(_dt, _acc_1, _gyr_1)

    # Evaluate residuals between system states and preintegration
    def evaluate(self, Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj):
        residuals = torch.zeros(15)

        # This part would involve matrix multiplication and updates, similar to the C++ code
        dp_dba = self.jacobian[O_P:O_P+3, O_BA:O_BA+3]
        dp_dbg = self.jacobian[O_P:O_P+3, O_BG:O_BG+3]
        dq_dbg = self.jacobian[O_R:O_R+3, O_BG:O_BG+3]
        dv_dba = self.jacobian[O_V:O_V+3, O_BA:O_BA+3]
        dv_dbg = self.jacobian[O_V:O_V+3, O_BG:O_BG+3]

        dba = Bai - self.linearized_ba
        dbg = Bgi - self.linearized_bg

        corrected_delta_q = torch.matmul(self.delta_q, Utils.Quateriond(torch.matmul(dq_dbg, dbg)))
        corrected_delta_v = self.delta_v + torch.matmul(dv_dba, dba) + torch.matmul(dv_dbg, dbg)
        corrected_delta_p = self.delta_p + torch.matmul(dp_dba, dba) + torch.matmul(dp_dbg, dbg)

        ### Residuals ###
        # residual p
        residuals[O_P:O_P+3] = \
            torch.matmul(Qi.T, (0.5 * G * self.sum_dt**2 + Pj - Pi - Vi * self.sum_dt)) - corrected_delta_p
        
        # residual q
        # TODO: test rotation
        residuals[O_R:O_R+3] = \
            2 * (torch.matmul(corrected_delta_q.inverse(), torch.matmul(Qi.inverse(), Qj))).vec()
        
        # residual v
        residuals[O_V:O_V+3] = \
            torch.matmul(Qi.inverse(), (G * self.sum_dt + Vj - Vi)) - corrected_delta_v
        
        # residual bias
        residuals[O_BA:O_BA+3] = Baj - Bai
        residuals[O_BG:O_BG+3] = Bgj - Bgi

        return residuals

    def checkJacobian(self):
        pass
    
    def checkIMUpropagation(self,_dt, _acc_1, _gyr_1):
        self.dt = _dt
        self.acc_1 = _acc_1
        self.gyr_1 = _gyr_1
        result_delta_p, result_delta_q, result_delta_v, \
            result_linearized_ba, result_linearized_bg = \
            self.midPointIntegration(_dt, self.acc_0, self.gyr_0, _acc_1, _gyr_1, \
                                    self.delta_p, self.delta_q, self.delta_v,
                                    self.linearized_ba, self.linearized_bg, True)

        # checkJacobian
        self.checkJacobian()
        
        # Update system variables
        self.delta_p = result_delta_p
        self.delta_q = result_delta_q
        self.delta_v = result_delta_v
        self.linearized_ba = result_linearized_ba
        self.linearized_bg = result_linearized_bg
        self.delta_q = Utils.NormalizeR(self.delta_q)
        self.sum_dt += self.dt
        
        print("delta p:", self.delta_p)
        print("delta q (as rotation matrix):", self.delta_q)
        print("delta v: ", self.delta_v)
        print("ba: ", self.linearized_ba)
        print("bg: ", self.linearized_bg)
        
