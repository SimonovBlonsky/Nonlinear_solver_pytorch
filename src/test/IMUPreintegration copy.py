import torch
from ..lietorch import SO3
import Utils

class IMUIntegration:
    def __init__(self, ba, bg):
        
        # raw data from IMU
        self.dt_buf_ = []
        self.acc_buf_ = []
        self.gyr_buf_ = []
        
        # pre-integrated IMU measurements
        self.sum_dt_ = 0
        self.delta_r_ = SO3.Identity(1, device="cuda")
        self.delta_v_ = torch.zeros(3, dtype=torch.float, device="cuda")
        self.delta_p_ = torch.zeros(3, dtype=torch.float, device="cuda")
        
        # gravity, biases
        self.gravity_ = 9.807
        self.bg_ = bg
        self.ba_ = ba
        
        # jacobians w.r.t ba and bg
        self.dr_dbg_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
        self.dv_dbg_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
        self.dv_dba_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
        self.dp_dbg_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
        self.dp_dba_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")

        # noise propagation
        self.covariance_measurement_ = torch.zeros(9, 9, dtype=torch.float, device="cuda")
        self.covariance_random_walk_ = torch.zeros(6, 6, dtype=torch.float, device="cuda")
        self.A_ = torch.zeros(9, 9, dtype=torch.float, device="cuda")
        self.B_ = torch.zeros(9, 6, dtype=torch.float, device="cuda")

        # raw noise of imu measurement
        self.noise_measurement_ = torch.zeros(6, 6, dtype=torch.float, device="cuda")
        self.noise_random_walk_ = torch.zeros(6, 6, dtype=torch.float, device="cuda")

        # @brief accelerometer measurement noise standard deviation*/
        self.acc_noise_ = 0.2
        # @brief gyroscope measurement noise standard deviation*/
        self.gyr_noise_ = 0.02
        # @brief accelerometer bias random walk noise standard deviation*/
        self.acc_random_walk_ = 0.0002;
        # @brief gyroscope bias random walk noise standard deviation*/
        self.gyr_random_walk_ = 2.0e-5;
        
        self.initNoiseParams()
        
    def initNoiseParams(self):
        I_3x3 = torch.eye(3, dtype=torch.float, device="cuda")
        self.noise_measurement_[0:3, 0:3] = (self.acc_noise_ * self.acc_noise_) * I_3x3
        self.noise_measurement_[3:6, 3:6] = (self.gyr_noise_ * self.gyr_noise_) * I_3x3
        self.noise_random_walk_[0:3, 0:3] = (self.acc_random_walk_ * self.acc_random_walk_) * I_3x3
        self.noise_random_walk_[3:6, 3:6] = (self.gyr_random_walk_ * self.gyr_random_walk_) * I_3x3
    
    # propage pre-integrated measurements using raw IMU data
    def propagate(self, dt, acc, gyr):
        self.dt_buf_.append(dt)
        self.acc_buf_.append(acc)
        self.gyr_buf_.append(gyr)
        
        I_3x3 = torch.eye(3, dtype=torch.float, device="cuda")
        
        # update delta r, v, p
        dR = SO3.exp((gyr - self.bg_) * dt)
        self.delta_r_ = self.delta_r_ * dR
        self.delta_v_ += self.delta_r_ * (acc - self.ba_) * dt
        self.delta_p_ += self.delta_v_ * dt\
                        + 0.5 * (self.delta_r_ * (acc - self.ba_) * dt * dt)

        # update jacobians w.r.t ba and bg
        self.dr_dbg_ -= self.delta_r_.inv().matrix() * Utils.JacobianR((gyr - self.bg_) * dt)
        self.dv_dba_ -= self.delta_r_.matrix() * dt
        self.dv_dbg_ -= self.delta_r_.matrix() * Utils.hat(acc - self. ba_) * self.dr_dbg_ * dt
        self.dp_dba_ += self.dv_dba_ * dt - 0.5 * self.delta_r_.matrix() * dt * dt
        self.dp_dbg_ += self.dv_dbg_ * dt - 0.5 * self.delta_r_.matrix() *\
            Utils.hat(acc - self.ba_) * self.dr_dbg_ * dt * dt
        
        # propagate noise
        self.A_[0:3, 0:3] = dR.inv().matrix()
        self.B_[0:3, 0:3] = Utils.JacobianR(dR.log())
        
        self.A_[3:6, 0:3] = -self.delta_r_.matrix() * Utils.hat(acc - self.ba_) * dt
        self.A_[3:6, 3:6] = I_3x3
        self.B_[3:6, 3:6] = self.delta_r_.matrix() * dt
        
        self.A_[6:9, 0:3] = -0.5 * self.delta_r_.matrix() * Utils.hat(acc - self.ba_) * dt * dt
        self.A_[6:9, 3:6] = I_3x3 * dt
        self.A_[6:9, 6:9] = I_3x3
        self.B_[6:9, 3:6] = 0.5 * self.delta_r_.matrix() * dt * dt
        
        self.covariance_measurement_ = self.A_ * self.covariance_measurement_ * self.A_.T\
            + self.B_ * self.noise_measurement_ * self.B_.T
        
    # according to pre-integration
    # when bias is updated, pre-integration should also be updated using first-order expansion of ba and bg
    def correct(self, delta_ba, delta_bg):
        self.delta_r_ = self.delta_r_ * SO3.exp(self.dr_dbg_ * delta_bg)
        self.delta_v_ += self.dv_dba_ * delta_ba + self.dv_dbg_ * delta_bg
        self.delta_p_ += self.dp_dba_ * delta_ba + self.dp_dbg_ * delta_bg
    
    # if bias is updated by a large value, redo the propagation
    def repropagate(self):
        dt = self.dt_buf_
        acc_buf = self.acc_buf_
        gyr_buf = self.gyr_buf_
        self.reset()
        for i in range(len(dt)): self.propagate(dt[i], acc_buf[i], gyr_buf[i])
    
    def setBiasG(self, bg): self.bg_ = bg
    def setBiasA(self, ba): self.ba_ = ba
    
    # Reset measurements and jacobians. Biases will not be reset
    def reset(self):
        self.sum_dt_ = 0
        self.delta_r_ = SO3.Identity(1, device="cuda")
        self.delta_v_ = torch.zeros(3, dtype=torch.float, device="cuda")
        self.delta_p_ = torch.zeros(3, dtype=torch.float, device="cuda")
        
        self.dr_dbg_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
        self.dv_dbg_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
        self.dv_dba_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
        self.dp_dbg_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")
        self.dp_dba_ = torch.zeros(3, 3, dtype=torch.float, device="cuda")

        self.covariance_measurement_ = torch.zeros(9, 9, dtype=torch.float, device="cuda")
        self.covariance_random_walk_ = torch.zeros(6, 6, dtype=torch.float, device="cuda")
        self.A_ = torch.zeros(9, 9, dtype=torch.float, device="cuda")
        self.B_ = torch.zeros(9, 6, dtype=torch.float, device="cuda")
    
    def getJacobians(self, dr_dbg, dv_dbg, dv_dba, dp_dbg, dp_dba):
        dr_dbg = self.dr_dbg_
        dv_dbg = self.dv_dbg_
        dv_dba = self.dv_dba_
        dp_dbg = self.dp_dbg_
        dp_dba = self.dp_dba_
    
    def getDrDbg(self): return self.dr_dbg_
    def getCovarianceMeasurement(self): return self.covariance_measurement_
    def getCovarianceRandomWalk(self): return self.covariance_random_walk_
    def getSumDt(self): return self.sum_dt_
    def getDeltaRVP(self): return self.delta_r_, self.delta_v_, self.delta_p_
    
    def getDr(self): return self.delta_r_
    def getDv(self): return self.delta_v_
    def getDp(self): return self.delta_p_