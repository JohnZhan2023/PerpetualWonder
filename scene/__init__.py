###
# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
# All rights reserved.
###
import random

from scene.dataset_readers import readDataInfo
from scene.gaussian_model import GaussianModel


class Scene:
    def __init__(self, traindata, gaussians: GaussianModel, opt , is_sky: bool = False, is_obj: bool = False, is_obj_init: bool = False):
        self.traindata = traindata
        self.gaussians = gaussians
        
        info = readDataInfo(traindata, opt.white_background)
        # random.shuffle(info.train_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        self.train_cameras = info.train_cameras

        if 'gaussians' in traindata.keys(): 
            self.gaussians.create_from_gaussians(info.point_cloud, self.cameras_extent, traindata['gaussians'])
        else:
            self.gaussians.create_from_pcd(info.point_cloud, self.cameras_extent, is_sky=is_sky, is_obj_init=is_obj_init)
        self.gaussians.training_setup(opt, is_obj=is_obj, is_obj_init=is_obj_init)

    def getTrainCameras(self):
        return self.train_cameras