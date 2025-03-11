import os
import cv2
import numpy as np
import re
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain
import math

def imu_stream(datapath, tstamp_scale):
    imu_all = np.loadtxt(datapath, delimiter=',')
    imu_all[:,0] /= tstamp_scale    # timestamp scaling
    imu_all[:, 1:4] *= 180 / math.pi    # omega
    imu_dict = dict(zip(imu_all[:,0], imu_all[:,1:]))
    return imu_dict
    #for t, data in imu_dict.items():
    #    queue.put((t, data))
    #queue.put((-1, data))
    

def image_stream(queue, imagedir, imagestamp, calib, stride, skip=0, dataset='EuRoC',imu=False):
    """ image generator """
    
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    
    if dataset == "EuRoC":
        img_exts = ["*.png", "*.jpeg", "*.jpg"]
        image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
        image_stamps = np.loadtxt(imagestamp, str, delimiter=',')
        image_dict = dict(zip(image_stamps[:,1], image_stamps[:,0]))
    elif dataset == "ETH3D":
        img_exts = ["*.png", "*.jpeg", "*.jpg"]
        image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
        image_stamps = np.loadtxt(imagestamp, str, delimiter=' ')
        image_dict = dict(zip(image_stamps[:,1],image_stamps[:,0]))
    elif dataset == "kitti":
        img_exts = ["*.png", "*.jpeg", "*.jpg"]
        image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
        image_stamps = np.loadtxt(imagestamp, str, delimiter=' ')
        image_dict = dict(zip(image_stamps[:,1], image_stamps[:,0]))
        
    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
        
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
        ### modified for V-I alignment ###
        if imu:
            if dataset == "EuRoC":
                tv = float(image_dict[imfile.name]) /1e9
            elif dataset == "ETH3D":
                tv = float(image_dict['rgb/'+ imfile.name])
            queue.put((tv, image, intrinsics))
        else:
            queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))