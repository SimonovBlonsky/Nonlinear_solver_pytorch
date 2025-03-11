from TestIMUintegration import TestIMUintegration
from IntegrationBase import IntegrationBase
from IMUstream import imu_stream
import numpy as np
import torch
from multiprocessing import Process, Queue

datapath = "/data/datasets/EuRoC_mav/MH_05_difficult/mav0/imu0/data.csv"

def test():
    """
    queue = Queue(maxsize=8)
    reader = Process(target=imu_stream, args=(queue, datapath, 1e9))
    reader.start()
    t, data = queue.get()
    testIMU = TestIMUintegration()
    """
    current_time = -1
    dx = 0
    dy = 0
    dz = 0
    rx = 0
    ry = 0
    rz = 0
    
    estimator = None
    imu_dict = imu_stream(datapath, 1e9)
    test_items = dict(list(imu_dict.items())[:8])
    
    for t, data in test_items.items():
        if estimator is None:
            estimator = TestIMUintegration()
        if current_time < 0: current_time = t
        dt = t - current_time
        dx = data[0].astype(np.float32)
        dy = data[1].astype(np.float32)
        dz = data[2].astype(np.float32)
        rx = data[3].astype(np.float32)
        ry = data[4].astype(np.float32)
        rz = data[5].astype(np.float32)
        
        acc = torch.tensor([dx, dy, dz])
        gyr = torch.tensor([rx, ry, rz])
        estimator.processIMU(dt, acc, gyr)
        estimator.processImage()

def testIMUpropagation():
    _dt = 0.005000114440917969
    _acc_1 = torch.tensor([-0.2400,  1.2800,  4.3600])
    _gyr_1 = torch.tensor([ 9.0139, -0.0572, -3.8001])

    acc_0 = torch.tensor([0.0400, 1.0800, 4.4800])
    gyr_0 = torch.tensor([ 9.0875, -0.0572, -3.8246])
    ba = torch.zeros(3)
    bg = torch.zeros(3)
    integrationBase = IntegrationBase(acc_0, gyr_0, ba, bg)
    integrationBase.checkIMUpropagation(_dt, _acc_1, _gyr_1)

if __name__ == "__main__":
    test()