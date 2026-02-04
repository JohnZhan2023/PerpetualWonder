import numpy as np
import torch
import struct
import os
import json
from toolkit.graphics import getWorld2View2, getProjectionMatrix

class Camera:
    def __init__(self, img_id, load_path=None, camtoworld=None, K=None, width=1280, height=704):
        if load_path is not None:
            data = np.load(load_path)
            self.K = torch.from_numpy(data["K"]).float()
            self.camtoworld = torch.from_numpy(data["camtoworld"]).float()
        elif camtoworld is not None and K is not None:
            self.K = K.clone() if isinstance(K, torch.Tensor) else torch.from_numpy(K).float()
            self.camtoworld = camtoworld.clone() if isinstance(camtoworld, torch.Tensor) else torch.from_numpy(camtoworld).float()
        else:
            raise ValueError("Camera must provide load_path, or camtoworld and K")
        self.worldtocam = torch.linalg.inv(self.camtoworld)
        self.img_id = img_id
        self.width = width
        self.height = height
        self.znear = 0.2
        self.zfar = 200.0
        self.K_original = self.K.clone()
        self.device = torch.device("cuda")


    def set_resolution(self, width, height):
        # original resolution from K
        original_width = self.K[0, 2] * 2
        original_height = self.K[1, 2] * 2
        
        scale_x = width / original_width
        scale_y = height / original_height
        
        scale = min(scale_x, scale_y)
        self.K[0, 0] = self.K_original[0, 0] * scale  # fx
        self.K[1, 1] = self.K_original[1, 1] * scale  # fy
        
        original_center_x = original_width / 2
        original_center_y = original_height / 2
        original_offset_x = self.K_original[0, 2] - original_center_x
        original_offset_y = self.K_original[1, 2] - original_center_y
    
        new_center_x = width / 2
        new_center_y = height / 2
        
        self.K[0, 2] = new_center_x + original_offset_x * scale  # cx
        self.K[1, 2] = new_center_y + original_offset_y * scale  # cy
        
        self.width = width
        self.height = height

    def to(self, device):
        self.K = self.K.to(device)
        self.K_original = self.K_original.to(device)
        self.camtoworld = self.camtoworld.to(device)
        self.worldtocam = self.worldtocam.to(device)
        return self

    def get_graphics_camera(self):
        self.fx, self.fy, self.cx, self.cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        self.fovx = 2 * np.arctan(self.width / (2 * self.fx))
        self.fovy = 2 * np.arctan(self.height / (2 * self.fy))
        
        w2c = self.worldtocam.cpu().numpy()
        self.R = np.transpose(w2c[:3, :3]) 
        self.T = w2c[:3, 3]
        
        viewmatrix = torch.tensor(getWorld2View2(self.R, self.T)).transpose(0, 1).cuda()

        projmatrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fovx, fovY=self.fovy).transpose(0,1).cuda()
        
        full_proj_transform = (viewmatrix.unsqueeze(0).bmm(projmatrix.unsqueeze(0))).squeeze(0)
        campos = viewmatrix.inverse()[3, :3]

        return viewmatrix, full_proj_transform, self.fovx, self.fovy, campos

def transform_matrix_to_str(matrix):
    # input is the camtoworld matrix
    matrix[:3, 3] *= 100
    matrix[:3, 1] *= -1.
    matrix = matrix[:, [2, 0, 1, 3]].T
    # turn to string
    matrix_str = ""
    for i in range(4):
        row = matrix[i]
        row_str = "[" + " ".join([f"{val:.6f}" for val in row]) + "] "
        matrix_str += row_str
    return matrix_str




def load_cameras_from_path(camera_path):

    import numpy as np
    import json

    with open(camera_path, "r") as f:
        camera_dict = json.load(f)

    camera_path = camera_dict["camera_path"]
    width = camera_dict["render_width"]
    height = camera_dict["render_height"]
    from nerfview import CameraState
    import viser.transforms as tf

    poses = []
    intrinsics = []
    for idx, cam_info in enumerate(camera_path):
        pose = tf.SE3.from_matrix(
            np.array(cam_info["camera_to_world"]).reshape(4, 4)
        )
        pose = tf.SE3.from_rotation_and_translation(
            pose.rotation() @ tf.SO3.from_x_radians(np.pi),
            pose.translation() * 10,
        )
        fov = cam_info["fov"]
        # turn fov to radians
        fov = np.deg2rad(fov)
        aspect = cam_info["aspect"]

        camera_state = CameraState(
            c2w=pose.as_matrix(),
            fov=fov,
            aspect=aspect,
        )
        pose = camera_state.c2w
        intrinsics.append(camera_state.get_K((width, height)))
        poses.append(pose)
    return poses, intrinsics

