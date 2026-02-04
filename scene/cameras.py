#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np

import torch
from torch import nn

from toolkit.graphics import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, 
                 image=torch.zeros([3, 512, 512]),
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 no_loss_mask=None,
                 gt_alpha_mask=None,
                 image_name=None,
                 colmap_id=None,
                 uid=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        if no_loss_mask is None:
            no_loss_mask = torch.zeros_like(self.original_image).bool()  # Do not remove image
        self.no_loss_mask = no_loss_mask.to(self.data_device)
        self.image_width = float(self.original_image.shape[2])
        self.image_height = float(self.original_image.shape[1])

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, int(self.image_height), int(self.image_width)), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = float(scale)

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale), dtype=torch.float32).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        self.focal_y = float(self.image_height) / (2.0 * tan_fovy)
        self.focal_x = float(self.image_width) / (2.0 * tan_fovx)
        
        # 确保所有矩阵都是float类型
        self.world_view_transform = self.world_view_transform.float()
        self.projection_matrix = self.projection_matrix.float()
        self.full_proj_transform = self.full_proj_transform.float()
        self.camera_center = self.camera_center.float()


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

