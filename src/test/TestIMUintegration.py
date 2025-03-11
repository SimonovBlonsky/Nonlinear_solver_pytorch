# Example usage
import torch
from dpvo.viba.IntegrationBase import IntegrationBase
import Utils
import LossFunction

class TestIMUintegration():
    def __init__(self):
        
        # Init params
        self.frame_count = 0
        self.WINDOW_SIZE = 8
        self.first_imu = False
        
        # IMU-cam extrinsics (not used)
        # self.tic = TIC
        # self.ric = RIC
        
        self.acc_0 = 0.0
        self.gyr_0 = 0.0
        
        # buffer
        self.dt_buf = []
        self.linear_acceleration_buf = []
        self.angular_velocity_buf = []
        
        # States
        # len: WINDOW_SIZE + 1
        self.Ps = []
        self.Vs = []
        self.Rs = []
        self.Bas = []
        self.Bgs = []
        self.g = torch.zeros(3)
        
        # Store preintegrations
        self.preintegrations = []
        self.tmp_preintegration = None
        
        # Images
        # self.all_image_frame[1.0] = ImageFrame()
        self.all_image_frame = {}
        
        # Image feature manager
        self.f_manager = None
        
    def clearState(self):
        for i in range(self.WINDOW_SIZE + 1):
            self.Rs[i] = torch.eye(3)
            self.Ps[i] = torch.zeros(3)
            self.Vs[i] = torch.zeros(3)
            self.Bas[i] = torch.zeros(3)
            self.Bgs[i] = torch.zeros(3)
        
        self.dt_buf = []
        self.linear_acceleration_buf = []
        self.angular_velocity_buf = []
        self.preintegrations = []
        
        for key, image_frame in self.all_image_frame.items():
            if image_frame.pre_integration is not None:
                image_frame.pre_integration = None
        

    def processIMU(self, dt, linear_acceleration, angular_velocity):
        """
        t0     t1    t2     t[frame_count]
           pi0    pi1   pi2             pi[frame_count]
        """
        # Process the first imu data
        if not self.first_imu:
            self.first_imu = True
            self.acc_0 = linear_acceleration
            self.gyr_0 = angular_velocity
        
        # If current frame_count don't have a preintegration, create one
        if not self.preintegrations[self.frame_count]:
            self.preintegrations[self.frame_count] = \
                IntegrationBase(self.acc_0, self.gyr_0, self.Bas[self.frame_count], self.Bgs[self.frame_count])
        
        # Process IMU
        if self.frame_count != 0:
            # set preintegration and propagate, update jacobian
            self.preintegrations[self.frame_count].push_back(dt, linear_acceleration, angular_velocity)
            self.tmp_preintegration.push_back(dt, linear_acceleration, angular_velocity)
            
            self.dt_buf[self.frame_count].push_back(dt)
            self.linear_acceleration_buf[self.frame_count].push_back(linear_acceleration)
            self.angular_velocity_buf[self.frame_count].push_back(angular_velocity)
            
            # Update states using mid-point measurements
            j = self.frame_count
            un_acc_0 = self.Rs[j] * (self.acc_0 - self.Bas[j]) - self.g
            un_gyr = 0.5 * (self.gyr_0 + angular_velocity) - self.Bgs[j]
            # TODO: test deltaQ
            self.Rs[j] *= Utils.deltaQ(un_gyr * dt).toRotationMatrix()
            un_acc_1 = self.Rs[j] * (linear_acceleration - self.Bas[j]) - self.g
            un_acc = 0.5 * (un_acc_0 + un_acc_1)
            self.Ps[j] += dt * self.Vs[j] + 0.5 * dt * dt * un_acc
            self.Vs[j] += dt * un_acc
            
        self.acc_0 = linear_acceleration
        self.gyr_0 = angular_velocity
    
    def processImage(self):
        pass
    
    def problemSolve(self):
        lossfunction = LossFunction.CauchyLoss(1.0)
        # TODO
    
    def sildeWindow(self):
        pass
        # TODO