'''
For the simplicity and efficiency of the open-source version, all the cases provided here have only one subgaussian, and the particle radius is set to infinity.
'''
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import List

def load_physical_params(phys_dir: str, frame_id: int) -> List[torch.Tensor]:
    """
    Load physical_params as foreground Gaussian points.
    Supports multiple entities and returns parameter list for each entity.
    
    Returns:
        List[torch.Tensor]: Parameters for each entity, each element has shape [2, N, 3] (position and velocity)
    """
    all_params = []
    entity_idx = 0
    
    while True:
        entity_dir = os.path.join(phys_dir, f"entity_{entity_idx}")
        if not os.path.exists(entity_dir):
            break
        
        params_path = os.path.join(entity_dir, f"{frame_id:04d}.npy")
        if not os.path.exists(params_path):
            break
        
        params = np.load(params_path)
        if params.shape[0] == 3:
            params = params[None, ...]
        elif len(params.shape) == 2:
            params = params[None, ...]
        
        all_params.append(torch.from_numpy(params).float())
        entity_idx += 1
    
    if len(all_params) == 0:
        # If no entity directory found, try loading directly (compatible with old format)
        params_path = os.path.join(phys_dir, f"{frame_id:04d}.npy")
        params = np.load(params_path)
        if params.shape[0] == 3:
            params = params[None, ...]
        elif len(params.shape) == 2:
            params = params[None, ...]
        return [torch.from_numpy(params).float()]
    
    # Return list without concatenation
    return all_params

def analyze_tensor_stats(name, tensor):
    arr = tensor.detach().cpu().numpy()
    print(f"[{name}] shape: {arr.shape}, min: {arr.min():.4f}, max: {arr.max():.4f}, mean: {arr.mean():.4f}, std: {arr.std():.4f}")

class GaussianModel:
    ###### ------------------ Basic Gaussian Attributes ------------------ ######
    def __init__(self, sh_degree=3, sub_gaussian=0, init_scale=0.03, subgaussian_size_min=0.001, subgaussian_size_max=0.01, dt=0.01, simulation_id=None, config=None):
        """
        args:
            previous_gaussian : GaussianModel; We take all of its 3DGS particles, freeze them and use them for rendering only.
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.init_scale = init_scale  # Add init_scale parameter
        self.subgaussian_size_min = subgaussian_size_min
        self.subgaussian_size_max = subgaussian_size_max
        self.dt = float(dt)
        self.simulation_id = simulation_id
        self.config = config
        

        self.sub_gaussian = sub_gaussian

        self.setup_functions()

    def setup_functions(self):
        """Setup functions used by the model"""
        pass  # No activation functions needed anymore

    ###### ------------------ I/O ------------------ ######

    def load_normal_background(self, work_dir, T = 49, simulation_id=None, add_foreground=False):
        """加载普通的背景高斯（用于优化），同时加载前景并设置为透明
        
        Args:
            work_dir: 工作目录
        """
        self.num_frames = T
        self.work_dir = work_dir
        if self.config['scene_name'] in ["jar"]:
            emitter_representation = True
            background_gs_dir = os.path.join(work_dir, "stage1_reconstruction/3d/ckpts/ckpt_9999.pt")

        else:
            emitter_representation = False
            background_gs_dir = os.path.join(work_dir, "stage1_reconstruction/background/background.pt")
        background_gs = torch.load(background_gs_dir)['splats']
        

        foreground_gs_dir = os.path.join(work_dir, "stage1_reconstruction/foreground", "foreground.pt")
        foreground_gs = torch.load(foreground_gs_dir)['splats']
        # Merge background and foreground positions
        bg_xyz = background_gs['means'].cuda()
        fg_xyz = foreground_gs['means'].cuda()
        # Directly remove points above z_quarter
        rate = self.config['rate']
        z_values = fg_xyz[:, 2]
        sorted_z, _ = torch.sort(z_values)
        if rate < 0:
            rate_ = 1 + rate
        else:
            rate_ = rate
        idx = int(len(sorted_z) * rate_)
        z_quarter = sorted_z[idx-1].item()
        if rate>0:
            mask = z_values > z_quarter
        elif rate == 0:
            mask = z_values >= 10000.0
        else:
            mask = z_values < z_quarter

        fg_xyz = fg_xyz[mask]

        combined_xyz = torch.cat([bg_xyz, fg_xyz], dim=0)
        
        # Record original number of background Gaussian splats (excluding foreground)
        self._original_bg_count = bg_xyz.shape[0]
        # Spherical harmonics coefficients
        bg_shs = background_gs['sh0'].cuda()
        fg_shs = foreground_gs['sh0'].cuda()
        fg_shs = fg_shs[mask]
        # set the fg_shs to be the sky blue
        combined_shs = torch.cat([bg_shs, fg_shs], dim=0)
        
        # Scaling
        bg_scales = background_gs['scales'].cuda()
        fg_scales = foreground_gs['scales'].cuda()
        fg_scales = fg_scales[mask]
        combined_scales = torch.cat([bg_scales, fg_scales], dim=0)
        
        # Rotation
        bg_rotations = background_gs['quats'].cuda()
        fg_rotations = foreground_gs['quats'].cuda()
        fg_rotations = fg_rotations[mask]
        combined_rotations = torch.cat([bg_rotations, fg_rotations], dim=0)
        
        # Opacity: background normal, foreground set to transparent
        bg_opacities = background_gs['opacities'].cuda().unsqueeze(1)
        fg_opacities = foreground_gs['opacities'].cuda().unsqueeze(1)
        fg_opacities = fg_opacities[mask]
        combined_opacities = torch.cat([bg_opacities, fg_opacities], dim=0)
        
        # Time parameters: background normal, foreground set to invisible time
        num_bg_gaussians = bg_xyz.shape[0]
        num_fg_gaussians = fg_xyz.shape[0]
        
        bg_t0 = torch.ones(num_bg_gaussians, dtype=torch.float32, device=bg_xyz.device) * 24
        fg_t0 = torch.ones(num_fg_gaussians, dtype=torch.float32, device=fg_xyz.device) * (24)  # Negative time
        combined_t0 = torch.cat([bg_t0, fg_t0], dim=0)
        
        bg_duration = torch.full((num_bg_gaussians,), 49.0, dtype=torch.float32, device=bg_xyz.device)
        fg_duration = torch.full((num_fg_gaussians,), 49.0, dtype=torch.float32, device=fg_xyz.device)
        combined_duration = torch.cat([bg_duration, fg_duration], dim=0)
        
        # Create shared parameters
        if add_foreground:
            for frame_id in range(self.num_frames):
                setattr(self, f"t_xyz_bg_{frame_id:03d}", nn.Parameter(combined_xyz.clone(), requires_grad=True))
            
            self._shs_bg = nn.Parameter(combined_shs, requires_grad=True)
            self._scaling_bg = nn.Parameter(combined_scales, requires_grad=True)
            self._rotation_bg = nn.Parameter(combined_rotations, requires_grad=True)
            self._opacity_bg = nn.Parameter(combined_opacities, requires_grad=True)
            self._t0_bg = nn.Parameter(combined_t0, requires_grad=True)
            self._duration_bg = nn.Parameter(combined_duration, requires_grad=True)
            print(f"Background gaussians: {background_gs['means'].shape[0]}")
            print(f"Foreground gaussians: {foreground_gs['means'].shape[0]}")
        else:
            for frame_id in range(self.num_frames):
                setattr(self, f"t_xyz_bg_{frame_id:03d}", nn.Parameter(bg_xyz.clone(), requires_grad=True))
            
            self._shs_bg = nn.Parameter(bg_shs, requires_grad=True)
            self._scaling_bg = nn.Parameter(bg_scales, requires_grad=True)
            self._rotation_bg = nn.Parameter(bg_rotations, requires_grad=True)
            self._opacity_bg = nn.Parameter(bg_opacities, requires_grad=True)
            self._t0_bg = nn.Parameter(bg_t0, requires_grad=True)
            self._duration_bg = nn.Parameter(bg_duration, requires_grad=True)
            print(f"Background gaussians: {background_gs['means'].shape[0]}")
        self.bg_gs_num = bg_xyz.shape[0]

    def load_optimized_background(self, work_dir, T=49, round_num = 1):
        """加载优化后的背景高斯
        
        Args:
            work_dir: 工作目录
            T: 帧数
            round_num: 轮数
        """
        self.num_frames = T
        background_gs_dir = os.path.join(work_dir, "optimized_background", f"round_{round_num}", "background.pt")
            
        background_gs = torch.load(background_gs_dir)['splats']
        
        # Use last frame's xyz to initialize xyz for all frames
        last_frame_xyz = background_gs[f'frame_{T-1:03d}']['means'].cuda()
        for frame_id in range(T):
            setattr(self, f"t_xyz_bg_{frame_id:03d}", last_frame_xyz.clone())
        
        # Load shared parameters
        self._shs_bg = background_gs['sh0'].cuda()
        self._scaling_bg = background_gs['scales'].cuda()
        self._rotation_bg = background_gs['quats'].cuda()
        self._opacity_bg = background_gs['opacities'].cuda()
        
        # Load time parameters
        self._t0_bg = background_gs['t0'].cuda()
        self._duration_bg = background_gs['duration']


    def load_from_work_dir(self, work_dir):
        if self.config['scene_name'] in ["snow_man"]:
            gs_representation = True
        else:
            gs_representation = False

        self.work_dir = work_dir
        if self.config['scene_name'] in ["jar"]:
            first_frame_gs_raw = []
            self.num_entities = 1
        elif not gs_representation and self.config['scene_name'] != "tap":
            gs_dir = os.path.join(work_dir, "stage1_reconstruction/3d", "ckpts")
            first_frame_gs_raw = torch.load(os.path.join(gs_dir, "ckpt_9999_rank0.pt"))['splats']
            self.num_entities = len(self.config["simulator_config"]["entities"])
        else:
            gs_dir = os.path.join(work_dir, "stage1_reconstruction/foreground", "foreground.pt")
            first_frame_gs_raw = torch.load(gs_dir)['splats']
            self.num_entities = 1

        physical_params_dir = os.path.join(work_dir, "stage2_forwardpass/physical_params", f"round_1")
        self.num_frames = 49
        
        # 加载每个entity的初始物理参数
        physic_initial_points_list = load_physical_params(physical_params_dir, 0)  # List of [2, N, 3]
        self.physic_initial_points_list = [params[0] for params in physic_initial_points_list]  # List of [N, 3]
        self.entity_is_rigid = [entity.get("is_rigid_body", False) for entity in self.config["simulator_config"]["entities"]]
        # 为每个entity初始化参数
        for entity_idx in range(self.num_entities):
            print("process entity ", entity_idx)
            gaussians_per_particle = self.config["simulator_config"]["entities"][entity_idx].get("gaussians_per_particle", 1)
            name = self.config["simulator_config"]["entities"][entity_idx].get("type", "entity")
            emitter_representation = "emitter" in name
            # 为刚体entity添加变换参数
            if self.entity_is_rigid[entity_idx]:
                for frame_id in range(self.num_frames):
                    setattr(self, f"rigid_rotation_{entity_idx}_{frame_id:03d}", nn.Parameter(
                        torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"), requires_grad=True
                    ))
                    setattr(self, f"rigid_translation_{entity_idx}_{frame_id:03d}", nn.Parameter(
                        torch.zeros(3, device="cuda"), requires_grad=True
                    ))
            
            # 为每一帧初始化anchor_points和xyz
            for frame_id in range(self.num_frames):
                genesis_points_list = load_physical_params(physical_params_dir, frame_id*self.config['simulator_config']['vis_frequency'])
                genesis_points = genesis_points_list[entity_idx][0]  # [N, 3]
                if not emitter_representation:
                    # Get initial means
                    if f"anchor_points_{entity_idx}" in first_frame_gs_raw:
                        initial_means = first_frame_gs_raw[f"anchor_points_{entity_idx}"].cuda()
                    elif gs_representation:
                        initial_means = first_frame_gs_raw["means"].cuda()
                    else:
                        initial_means = self.physic_initial_points_list[entity_idx].cuda()
                    
                    t_xyz = initial_means + genesis_points.cuda() - self.physic_initial_points_list[entity_idx].cuda()
                    # t_xyz = initial_means
                    anchor_points = t_xyz.clone().detach().cuda()
                    anchor_points.requires_grad_(False)
                else:
                    # Emitter handling logic remains unchanged
                    t_xyz = genesis_points.cuda().repeat_interleave(gaussians_per_particle, dim=0)
                    torch.manual_seed(42)
                    self.meta_ball_size = self.config["simulator_config"]['particle_size']
                    random_offsets = torch.randn_like(t_xyz)
                    t_xyz = random_offsets
                    anchor_points = genesis_points.clone().detach().cuda().repeat_interleave(gaussians_per_particle, dim=0)
                    mask = anchor_points.norm(dim=-1) < 100
                    setattr(self, f"mask_{entity_idx}_{frame_id:03d}", mask)
                
                setattr(self, f"t_xyz_{entity_idx}_{frame_id:03d}", nn.Parameter(t_xyz.clone().detach(), requires_grad=True))
                setattr(self, f"anchor_points_{entity_idx}_{frame_id:03d}", anchor_points)
                
                # Initialize shared parameters only for the first frame
                if frame_id == 0:
                    if not emitter_representation and f"sh0_{entity_idx}" in first_frame_gs_raw:
                        setattr(self, f"scaling_{entity_idx}_{frame_id:03d}", nn.Parameter(first_frame_gs_raw[f"scales_{entity_idx}"].cuda(), requires_grad=True))
                        setattr(self, f"sh0_{entity_idx}_{frame_id:03d}", nn.Parameter(first_frame_gs_raw[f"sh0_{entity_idx}"].cuda(), requires_grad=True))
                        setattr(self, f"t_rotation_{entity_idx}_{frame_id:03d}", nn.Parameter(first_frame_gs_raw[f"quats_{entity_idx}"].cuda(), requires_grad=True))
                        num_gaussians = first_frame_gs_raw[f"opacities_{entity_idx}"].shape[0]
                        setattr(self, f"opacity_{entity_idx}_{frame_id:03d}", nn.Parameter(first_frame_gs_raw[f"opacities_{entity_idx}"].clone().cuda().unsqueeze(1), requires_grad=True))
                    elif gs_representation:
                        setattr(self, f"scaling_{entity_idx}_{frame_id:03d}", nn.Parameter(first_frame_gs_raw["scales"].cuda(), requires_grad=True))
                        setattr(self, f"sh0_{entity_idx}_{frame_id:03d}", nn.Parameter(first_frame_gs_raw["sh0"].cuda(), requires_grad=True))
                        setattr(self, f"t_rotation_{entity_idx}_{frame_id:03d}", nn.Parameter(first_frame_gs_raw["quats"].cuda(), requires_grad=True))
                        setattr(self, f"opacity_{entity_idx}_{frame_id:03d}", nn.Parameter(first_frame_gs_raw["opacities"].clone().cuda().unsqueeze(1), requires_grad=True))
                        num_gaussians = first_frame_gs_raw["means"].shape[0]
                    else:
                        # Emitter initialization logic
                        num_particles = self.physic_initial_points_list[entity_idx].shape[0]
                        total_gaussians = num_particles * gaussians_per_particle
                        base_scaling = torch.logit(torch.ones(total_gaussians, 3, device="cuda") * self.config["simulator_config"]['particle_size']/2.0)
                        setattr(self, f"scaling_{entity_idx}_{frame_id:03d}", nn.Parameter(base_scaling, requires_grad=True))
                        
                        sh0 = torch.zeros(total_gaussians, 1, 3, device="cuda")
                        preset_sh0 = self.config["simulator_config"]['entities'][entity_idx]['sh0']
                        sh0[..., 0] = preset_sh0[0]
                        sh0[..., 1] = preset_sh0[1]
                        sh0[..., 2] = preset_sh0[2]
                        setattr(self, f"sh0_{entity_idx}_{frame_id:03d}", nn.Parameter(sh0, requires_grad=True))
                        
                        quats = torch.zeros(total_gaussians, 4, device="cuda")
                        quats[..., 0] = 1.0
                        setattr(self, f"t_rotation_{entity_idx}_{frame_id:03d}", nn.Parameter(quats, requires_grad=True))
                        
                        opacities = torch.logit(torch.ones(total_gaussians, 1, device="cuda") * 0.99)
                        setattr(self, f"opacity_{entity_idx}_{frame_id:03d}", nn.Parameter(opacities, requires_grad=True))
                        num_gaussians = total_gaussians
                    
                    setattr(self, f"t0_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(torch.ones(num_gaussians, dtype=torch.float32, device="cuda")*24, requires_grad=True))
                    setattr(self, f"duration_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(torch.full((num_gaussians,), 49.0, dtype=torch.float32, device="cuda"), requires_grad=True))

        return True


    #     return True
    def load_from_work_dir_further(self, work_dir, round_num = 1):
        self.work_dir = work_dir
        # 加载上一轮的generative simulation结果
        print("load generative simulation from ", os.path.join(work_dir, "stage3_optimization", "generative_simulation", f"round_{round_num-1}.pt"))
        entities_data = torch.load(os.path.join(work_dir, "stage3_optimization", "generative_simulation", f"round_{round_num-1}.pt"))
        
        self.num_entities = len(self.config["simulator_config"]["entities"])
        # TEMPORAL FOR CAKE
        # self.num_entities = 1
        self.entity_is_rigid = [entity.get("is_rigid_body", False) for entity in self.config["simulator_config"]["entities"]]
        
        # 加载所有帧的physical params
        physical_params_dir = os.path.join(work_dir, "stage2_forwardpass", "physical_params", f"round_{round_num}")
        self.num_frames = 49
        
        # 加载每个entity的初始物理参数
        physic_initial_points_list = load_physical_params(physical_params_dir, 0)
        self.physic_initial_points_list = [params[0] for params in physic_initial_points_list]
        
        # 为每个entity初始化参数
        for entity_idx, entity_data in enumerate(entities_data):
            emitter_representation = "emitter" in self.config["simulator_config"]["entities"][entity_idx].get("type", "entity")
            scales = entity_data['scaling'].cuda()
            # scales = torch.logit(torch.ones_like(scales) * self.config["simulator_config"]["entities"][entity_idx]['init_scale'])
            rotation = F.normalize(entity_data['rotation'].cuda(), dim=-1)
            xyzs = entity_data['all_frames_xyz'][-1].cuda()
            opacities = entity_data['opacities'].cuda()
            # Set all points with low opacity to opaque
            
            shs = entity_data['shs'].cuda()     
            t0 = entity_data['t0'].cuda()
            duration = entity_data['duration'].cuda()
            offset_t0 = t0 - 49

            # If rigid body, initialize transformation parameters
            if self.entity_is_rigid[entity_idx]:
                for frame_id in range(self.num_frames):
                    setattr(self, f"rigid_rotation_{entity_idx}_{frame_id:03d}", nn.Parameter(
                        torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"), requires_grad=True
                    ))
                    setattr(self, f"rigid_translation_{entity_idx}_{frame_id:03d}", nn.Parameter(
                        torch.zeros(3, device="cuda"), requires_grad=True
                    ))
            
            # Set anchor points and means for each frame
            for frame_id in range(self.num_frames):
                genesis_points_list = load_physical_params(physical_params_dir, frame_id*8)
                genesis_points = genesis_points_list[entity_idx][0]
            
                initial_points = self.physic_initial_points_list[entity_idx]
                

                if emitter_representation:
                    gaussians_per_particle = self.config["simulator_config"]["entities"][entity_idx].get("gaussians_per_particle", 1)
                    anchor_points = genesis_points.cuda().repeat_interleave(gaussians_per_particle, dim=0)
                    self.meta_ball_size = self.config["simulator_config"]["entities"][entity_idx]['init_scale']
                    t_xyz = torch.randn_like(anchor_points)/100.0
                    # t_xyz = anchor_points.clone()
                    mask = anchor_points.norm(dim=-1) < 2
                    setattr(self, f"mask_{entity_idx}_{frame_id:03d}", mask)
                else:
                    t_xyz = xyzs + genesis_points.cuda() - initial_points.cuda()
                    anchor_points = t_xyz.clone().detach().cuda()
                
                setattr(self, f"t_xyz_{entity_idx}_{frame_id:03d}", nn.Parameter(t_xyz.clone().detach(), requires_grad=True))
                setattr(self, f"anchor_points_{entity_idx}_{frame_id:03d}", anchor_points)
                
                if frame_id == 0:
                    if not emitter_representation:
                        setattr(self, f"opacity_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(opacities.clone(), requires_grad=True))
                        setattr(self, f"t_rotation_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(rotation.clone(), requires_grad=True))
                        setattr(self, f"sh0_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(shs.clone(), requires_grad=True))
                        setattr(self, f"scaling_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(scales.clone(), requires_grad=True))
                        setattr(self, f"t0_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(offset_t0.clone(), requires_grad=True))
                        setattr(self, f"duration_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(duration.clone(), requires_grad=True))
                    else:
                        gaussians_per_particle = self.config["simulator_config"]["entities"][entity_idx].get("gaussians_per_particle", 1)
                        total_gaussians = anchor_points.shape[0]
                        sh0 = torch.zeros(total_gaussians, 1, 3, device="cuda")
                        preset_sh0 = self.config["simulator_config"]['entities'][entity_idx]['sh0']
                        sh0[..., 0] = preset_sh0[0]
                        sh0[..., 1] = preset_sh0[1]
                        sh0[..., 2] = preset_sh0[2]
                        setattr(self, f"scaling_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(scales.clone(), requires_grad=True))
                        setattr(self, f"sh0_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(sh0.clone(), requires_grad=True))
                        setattr(self, f"t_rotation_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(rotation.clone(), requires_grad=True))
                        setattr(self, f"opacity_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(opacities.clone(), requires_grad=True))
                        setattr(self, f"t0_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(offset_t0.clone(), requires_grad=True))
                        setattr(self, f"duration_{entity_idx}_{frame_id:03d}", torch.nn.Parameter(duration.clone(), requires_grad=True))
                    
        return True

###### ------------------ Training ------------------ ######

    def training_4d_setup(self, frames_num):
        """为每一帧设置优化参数
        
        Args:
            frames_num: 总帧数
        """

        # set all bg gaussians not to be optimized
        self._shs_bg.requires_grad_(False)
        self._scaling_bg.requires_grad_(False)
        self._rotation_bg.requires_grad_(False)
        self._opacity_bg.requires_grad_(False)
        self._t0_bg.requires_grad_(False)
        self._duration_bg.requires_grad_(False)
        # set bg xyz not to be optimized
        for t_idx in range(frames_num):
            getattr(self, f"t_xyz_bg_{t_idx:03d}").requires_grad_(False)
        
        optimize_params = []
        
        # 为每个entity设置优化参数
        for entity_idx in range(self.num_entities):
            for t_idx in range(frames_num):
                xyz = getattr(self, f"t_xyz_{entity_idx}_{t_idx:03d}")
                
                # 如果是刚体entity，不优化xyz坐标，而是优化刚体变换参数
                if self.entity_is_rigid[entity_idx]:
                    xyz.requires_grad_(False)
                    
                    # 添加刚体变换参数的优化
                    rigid_rotation = getattr(self, f"rigid_rotation_{entity_idx}_{t_idx:03d}")
                    rigid_translation = getattr(self, f"rigid_translation_{entity_idx}_{t_idx:03d}")
                    
                    optimize_params.extend([
                        {
                            "params": rigid_rotation,
                            "lr": 1e-3,
                            "name": f"rigid_rotation_{entity_idx}_{t_idx:03d}",
                        },
                        {
                            "params": rigid_translation,
                            "lr": 1e-3,
                            "name": f"rigid_translation_{entity_idx}_{t_idx:03d}",
                        }
                    ])
                    rigid_rotation.requires_grad_(True)
                    rigid_translation.requires_grad_(True)
                else:
                    # Non-rigid body entity, optimize xyz coordinates
                    optimize_params.append({
                        "params": xyz,
                        "lr": 2e-4,
                        # "lr": 1.0,
                        "name": f"xyz_{entity_idx}_{t_idx:03d}",
                    })
                    xyz.requires_grad_(True)

                # Set other parameters only for the first frame
                if t_idx == 0:
                    sh0 = getattr(self, f"sh0_{entity_idx}_{t_idx:03d}")
                    opacity = getattr(self, f"opacity_{entity_idx}_{t_idx:03d}")
                    t0 = getattr(self, f"t0_{entity_idx}_{t_idx:03d}")
                    duration = getattr(self, f"duration_{entity_idx}_{t_idx:03d}")
                    scaling = getattr(self, f"scaling_{entity_idx}_{t_idx:03d}")
                    optimize_params.extend([
                        {
                            "params": sh0,
                            "lr": 3e-2,
                            "name": f"sh0_{entity_idx}_{t_idx:03d}",
                        },
                        {
                            "params": opacity,
                            "lr": 3e-2,
                            "name": f"opacity_{entity_idx}_{t_idx:03d}",
                        },
                        {
                            "params": t0,
                            "lr": 1.0,
                            "name": f"t0_{entity_idx}_{t_idx:03d}",
                        },
                        {
                            "params": duration,
                            "lr": 1.0,
                            "name": f"duration_{entity_idx}_{t_idx:03d}",
                        },
                        {
                            "params": scaling,
                            "lr": 0,
                            "name": f"scaling_{entity_idx}_{t_idx:03d}",
                        }
                    ])
                    sh0.requires_grad_(True)
                    opacity.requires_grad_(True)
                    t0.requires_grad_(True)
                    duration.requires_grad_(True)
                    scaling.requires_grad_(True)


        self.optimizer = torch.optim.Adam(optimize_params, lr=0.008, eps=1e-15, betas=(0.0, 0.99))


    def training_bg_setup(self, frames_num):
        """Setup optimization parameters for background Gaussian points"""
        optimize_params = []
        
        # Add optimization parameters for xyz of each frame
        for t_idx in range(frames_num):
            xyz = getattr(self, f"t_xyz_bg_{t_idx:03d}")
            optimize_params.append(
                {"params": xyz, "lr": 0, "name": f"xyz_bg_{t_idx:03d}"}
            )
        
        # Add shared parameters
        optimize_params.extend([
            {"params": self._shs_bg, "lr": 3e-4, "name": "shs_bg"},
            {"params": self._scaling_bg, "lr": 1e-4, "name": "scaling_bg"},
            {"params": self._rotation_bg, "lr": 1e-4, "name": "rotation_bg"},
            {"params": self._opacity_bg, "lr": 1e-4, "name": "opacity_bg"},
            {"params": self._t0_bg, "lr": 0.5, "name": "t0_bg"},
            {"params": self._duration_bg, "lr": 0.5, "name": "duration_bg"},
        ])
        
        self.optimizer = torch.optim.Adam(optimize_params, lr=0.008, eps=1e-15, betas=(0.0, 0.99))
        return True



###### ------------------ Getters ------------------ ######

    def get_rotation(self, t=0):
        """获取指定帧的旋转，遍历所有entity"""
        all_rotations = []
        for entity_idx in range(self.num_entities):
            rotation = getattr(self, f"t_rotation_{entity_idx}_{0:03d}")
            if hasattr(self, f"mask_{entity_idx}_{t:03d}"):
                mask = getattr(self, f"mask_{entity_idx}_{t:03d}")
                rotation = rotation[mask]
            all_rotations.append(rotation)
        
        rotation = torch.cat(all_rotations, dim=0)
        return rotation
    
    def get_xyz(self, t=0):
        """Get Gaussian point positions for specified frame, iterate over all entities
        
        Returns:
            torch.Tensor: Actual positions of foreground points
        """
        all_xyz = []
        for entity_idx in range(self.num_entities):
            xyz = getattr(self, f"t_xyz_{entity_idx}_{t:03d}")  # [N, 3]

            
            if hasattr(self, 'meta_ball_size'):
                anchor_points = getattr(self, f"anchor_points_{entity_idx}_{t:03d}")
                xyz = anchor_points + torch.tanh(xyz) * self.meta_ball_size
            
            if self.entity_is_rigid[entity_idx]:
                xyz = self.apply_rigid_transform(xyz, entity_idx, t)
            
            if hasattr(self, f"mask_{entity_idx}_{t:03d}"):
                mask = getattr(self, f"mask_{entity_idx}_{t:03d}")
                xyz = xyz[mask]
            
            all_xyz.append(xyz)
        
        return torch.cat(all_xyz, dim=0)

    def apply_rigid_transform(self, xyz, entity_idx, t):
        """Apply rigid body transformation to xyz coordinates
        
        Args:
            xyz: Original xyz coordinates [N, 3]
            entity_idx: Entity index
            t: Time frame
            
        Returns:
            torch.Tensor: Transformed xyz coordinates
        """
        # Get rotation and translation parameters
        rotation_quat = getattr(self, f"rigid_rotation_{entity_idx}_{t:03d}")  # [4]
        translation = getattr(self, f"rigid_translation_{entity_idx}_{t:03d}")  # [3]
        
        # Normalize quaternion
        rotation_quat = F.normalize(rotation_quat, dim=0)
        
        # Convert quaternion to rotation matrix
        rotation_matrix = self.quat_to_rotation_matrix(rotation_quat)  # [3, 3]
        
        # Apply transformation: xyz_new = R * xyz + t
        xyz_transformed = torch.matmul(xyz, rotation_matrix.T) + translation.unsqueeze(0)
        
        return xyz_transformed
    
    def quat_to_rotation_matrix(self, quat):
        """Convert quaternion to rotation matrix
        
        Args:
            quat: Quaternion [w, x, y, z]
            
        Returns:
            torch.Tensor: 3x3 rotation matrix
        """
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        
        # Build rotation matrix
        rotation_matrix = torch.tensor([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ], device=quat.device, dtype=quat.dtype)
        
        return rotation_matrix

    def get_anchor_points(self, t=0):
        """Get anchor points for specified frame, iterate over all entities"""
        all_anchor_points = []
        for entity_idx in range(self.num_entities):
            anchor_points = getattr(self, f"anchor_points_{entity_idx}_{t:03d}")#getattr(self, f"anchor_points_0_000")
            if hasattr(self, f"mask_{entity_idx}_{t:03d}"):
                mask = getattr(self, f"mask_{entity_idx}_{t:03d}")
                anchor_points = anchor_points[mask]
            all_anchor_points.append(anchor_points)
        return torch.cat(all_anchor_points, dim=0)

    def get_features(self, t=0):
        """Get features (spherical harmonics coefficients) for specified frame, iterate over all entities"""
        all_features = []
        for entity_idx in range(self.num_entities):
            sh0 = getattr(self, f"sh0_{entity_idx}_{0:03d}")
            if hasattr(self, f"mask_{entity_idx}_{t:03d}"):
                mask = getattr(self, f"mask_{entity_idx}_{t:03d}")
                sh0 = sh0[mask]
            all_features.append(sh0)
        
        features = torch.cat(all_features, dim=0)
    
        return features
    
    def get_opacity(self, t=0):
        """Get opacity for specified frame, iterate over all entities
        
        Args:
            t: Current time point
            
        Returns:
            torch.Tensor: Final opacity
        """
        all_opacities = []
        for entity_idx in range(self.num_entities):
            opacity = getattr(self, f"opacity_{entity_idx}_{0:03d}")
            spatial_opacity = torch.sigmoid(opacity).squeeze()
            
            t0 = getattr(self, f"t0_{entity_idx}_{0:03d}")
            duration = getattr(self, f"duration_{entity_idx}_{0:03d}")
            t_current = torch.tensor(t, dtype=torch.float32, device=opacity.device)
            temporal_opacity = torch.exp(-0.5 * ((t_current - t0) / duration) ** 2)
            obj_opacity = spatial_opacity * temporal_opacity

            if hasattr(self, f"mask_{entity_idx}_{t:03d}"):
                mask = getattr(self, f"mask_{entity_idx}_{t:03d}")
                obj_opacity = obj_opacity[mask]

            all_opacities.append(obj_opacity)
        
        return torch.cat(all_opacities, dim=0)
    
    def get_scaling(self, t=0):
        """Get scaling for specified frame, iterate over all entities
        
        Args:
            t: Time frame
        """
        all_scalings = []
        for entity_idx in range(self.num_entities):
            scaling = getattr(self, f"scaling_{entity_idx}_{0:03d}")
            scaling = torch.exp(scaling)
            if scaling.shape[1] == 1:
                scaling = scaling.repeat(1, 3)
            
            if hasattr(self, 'meta_ball_size'):
                scaling = torch.clamp(scaling, max=self.meta_ball_size)
            
            if hasattr(self, f"mask_{entity_idx}_{t:03d}"):
                mask = getattr(self, f"mask_{entity_idx}_{t:03d}")
                scaling = scaling[mask]
            
            all_scalings.append(scaling)
        
        scaling = torch.cat(all_scalings, dim=0)

        
        return scaling

    def get_xyz_bg(self, t=0):
        """Get background Gaussian point positions
        
        Args:
            t: Time frame
        Returns:
            torch.Tensor: Background Gaussian positions at current time
        """
        xyz = getattr(self, f"t_xyz_bg_{t:03d}")
        return xyz

    def get_rotation_bg(self, t=0):
        """Get background Gaussian point rotation"""
        return self._rotation_bg
    
    def get_scaling_bg(self, t=0):
        """Get background Gaussian point scaling"""
        scaling_bg = self._scaling_bg
        scaling_bg = torch.exp(scaling_bg).float()
        if scaling_bg.shape[1] == 1:
            scaling_bg = scaling_bg.repeat(1, 3)
        return scaling_bg
    
    def get_opacity_bg(self, t=0):
        """Get background Gaussian point opacity, including spatial and temporal opacity
        
        Args:
            t: Current time point
            
        Returns:
            torch.Tensor: Final opacity, obtained by multiplying spatial opacity and temporal opacity
        """
        # Spatial opacity
        spatial_opacity = torch.sigmoid(self._opacity_bg.squeeze()).float()
        
        # Temporal opacity: σ(t) = exp(-0.5 * ((t - μt) / s)^2)
        t_current = torch.tensor(t, dtype=torch.float32, device=self._opacity_bg.device)
        temporal_opacity = torch.exp(-0.5 * ((t_current - self._t0_bg) / self._duration_bg) ** 2)
        
        # Final opacity is the product of temporal opacity and spatial opacity
        return spatial_opacity * temporal_opacity
    
    def get_features_bg(self, t=0):
        """Get background Gaussian point features
        """
        return self._shs_bg.float()

    def save_generative_simulation(self, round_num = 1, semi_round = True):
        """
        按entity分开保存所有帧的xyz和其他参数
        """
        T = self.num_frames
        os.makedirs(os.path.join(self.work_dir, "stage3_optimization", "generative_simulation"), exist_ok=True)
        
        # 为每个entity分别保存
        entities_data = []
        for entity_idx in range(self.num_entities):
            # 收集所有帧的xyz
            all_frames_xyz = []
            for frame_id in range(T):
                xyz = getattr(self, f"t_xyz_{entity_idx}_{frame_id:03d}").clone().detach()
                if hasattr(self, 'meta_ball_size'):
                    anchor_points = getattr(self, f"anchor_points_{entity_idx}_{frame_id:03d}").clone().detach()
                    xyz_current = anchor_points + torch.tanh(xyz) * self.meta_ball_size
                else:
                    xyz_current = xyz
                
                # 如果是刚体，应用变换
                if self.entity_is_rigid[entity_idx]:
                    xyz_current = self.apply_rigid_transform(xyz_current, entity_idx, frame_id)
                
                all_frames_xyz.append(xyz_current)
            
            # 获取原始参数
            raw_opacity = getattr(self, f"opacity_{entity_idx}_{0:03d}").clone().detach()
            raw_scaling = getattr(self, f"scaling_{entity_idx}_{0:03d}").clone().detach()
            raw_rotation = getattr(self, f"t_rotation_{entity_idx}_{0:03d}").clone().detach()
            raw_shs = getattr(self, f"sh0_{entity_idx}_{0:03d}").clone().detach()
            t0 = getattr(self, f"t0_{entity_idx}_{0:03d}").clone().detach()
            duration = getattr(self, f"duration_{entity_idx}_{0:03d}").clone().detach()
            
            # 计算combined_opacity
            spatial_opacity = torch.sigmoid(raw_opacity).squeeze(1)
            t_current = torch.tensor(T-1, dtype=torch.float32, device=raw_opacity.device)
            temporal_opacity = torch.exp(-0.5 * ((t_current - t0) / duration) ** 2)
            combined_opacity = spatial_opacity * temporal_opacity
            
            entity_data = {
                'all_frames_xyz': all_frames_xyz,  # 包含所有帧的xyz列表
                'opacities': raw_opacity,
                'scaling': raw_scaling,
                'rotation': raw_rotation,
                'shs': raw_shs,
                't0': t0,
                'duration': duration,
                'combined_opacity': combined_opacity,
            }
            
            # 如果是刚体，保存变换参数
            if self.entity_is_rigid[entity_idx]:
                rigid_rotations = []
                rigid_translations = []
                for frame_id in range(T):
                    rigid_rotation = getattr(self, f"rigid_rotation_{entity_idx}_{frame_id:03d}").clone().detach()
                    rigid_translation = getattr(self, f"rigid_translation_{entity_idx}_{frame_id:03d}").clone().detach()
                    rigid_rotations.append(rigid_rotation)
                    rigid_translations.append(rigid_translation)
                entity_data['rigid_rotations'] = rigid_rotations
                entity_data['rigid_translations'] = rigid_translations
            
            entities_data.append(entity_data)
        
        # 保存所有entity的数据
        torch.save(entities_data, os.path.join(self.work_dir, "stage3_optimization", "generative_simulation", f"round_{round_num}.pt"))
        print("save generative simulation to ", os.path.join(self.work_dir, "stage3_optimization", "generative_simulation", f"round_{round_num}.pt"))
    #DEBUG
    def load_background(self, work_dir, round_num, add_foreground=True):


        """
        加载背景数据，优先加载特定round的背景，如果不存在则加载默认背景
        Args:
            round_num: 轮次编号
            add_foreground: 是否添加前景高斯点
        """
        self.work_dir = work_dir
        # 尝试加载特定round的背景
        round_background_path = os.path.join(self.work_dir, "stage3_optimization", "optimized_background", f"round_{round_num}", "background.pt")
        default_background_path = os.path.join(self.work_dir, "stage1_reconstruction", "background", "background.pt")
        
        if os.path.exists(round_background_path):
            background_path = round_background_path
            print(f"使用round_{round_num}的背景数据")
        elif os.path.exists(default_background_path):
            background_path = default_background_path
            print("使用默认背景数据")
        else:
            raise FileNotFoundError(f"找不到任何背景数据文件")
        
        background_gs = torch.load(background_path)['splats']
        
        # 加载前景数据（如果需要）
        if add_foreground:
            foreground_gs_dir = os.path.join(self.work_dir, "stage1_reconstruction", "foreground", "foreground.pt")
            foreground_gs = torch.load(foreground_gs_dir)['splats']
            
            # 合并背景和前景的位置
            for frame_id in range(49):
                if f"frame_{frame_id:03d}" in background_gs:
                    setattr(self, f"t_xyz_bg_{frame_id:03d}", background_gs[f'frame_{frame_id:03d}']['means'].cuda())
                else:
                    setattr(self, f"t_xyz_bg_{frame_id:03d}", background_gs['means'].cuda())
            fg_xyz = foreground_gs['means'].cuda()
            
            # Filter foreground points based on z values
            rate = self.config['rate']
            z_values = fg_xyz[:, 2]
            sorted_z, _ = torch.sort(z_values)
            if rate < 0:
                rate_ = 1 + rate
            else:
                rate_ = rate
            idx = int(len(sorted_z) * rate_)
            z_quarter = sorted_z[idx].item()
            if rate > 0:
                mask = z_values > z_quarter
            elif rate == 0:
                mask = z_values >= 10000.0
            else:
                mask = z_values < z_quarter
            mask = ~mask
            
            fg_xyz = fg_xyz[mask]
            for frame in range(49):
                if f"frame_{frame:03d}" in background_gs:
                    setattr(self, f"combined_xyz_{frame:03d}", torch.cat([background_gs[f'frame_{frame:03d}']['means'].cuda(), fg_xyz], dim=0))
                else:
                    setattr(self, f"combined_xyz_{frame:03d}", torch.cat([background_gs['means'].cuda(), fg_xyz], dim=0))
            
            # Record original number of background Gaussian splats
            self._original_bg_count = background_gs['scales'].shape[0]
            num_bg_gaussians = background_gs['scales'].shape[0]
            
            # Merge other attributes
            bg_shs = background_gs['sh0'].cuda()
            fg_shs = foreground_gs['sh0'].cuda()[mask]
            combined_shs = torch.cat([bg_shs, fg_shs], dim=0)
            
            bg_scales = background_gs['scales'].cuda()
            fg_scales = foreground_gs['scales'].cuda()[mask]
            combined_scales = torch.cat([bg_scales, fg_scales], dim=0)
            
            bg_rotations = background_gs['quats'].cuda()
            fg_rotations = foreground_gs['quats'].cuda()[mask]
            combined_rotations = torch.cat([bg_rotations, fg_rotations], dim=0)
            
            bg_opacities = background_gs['opacities'].cuda().squeeze()
            fg_opacities = foreground_gs['opacities'].cuda()[mask]
            combined_opacities = torch.cat([bg_opacities, fg_opacities], dim=0).unsqueeze(1)
            
            # Set time parameters
            num_fg_gaussians = fg_xyz.shape[0]
            
            bg_t0 = torch.ones(num_bg_gaussians, dtype=torch.float32, device="cuda") * 24
            fg_t0 = torch.ones(num_fg_gaussians, dtype=torch.float32, device="cuda") * 24
            combined_t0 = torch.cat([bg_t0, fg_t0], dim=0)
            
            bg_duration = torch.full((num_bg_gaussians,), 49.0, dtype=torch.float32, device="cuda")
            fg_duration = torch.full((num_fg_gaussians,), 49.0, dtype=torch.float32, device="cuda")
            combined_duration = torch.cat([bg_duration, fg_duration], dim=0)
            
            # Set attributes
            # if "means" in background_gs:
            #     for frame_id in range(49):
            #         setattr(self, f"t_xyz_bg_{frame_id:03d}", background_gs['means'].cuda())
            # else:
            for frame_id in range(49):
                setattr(self, f"t_xyz_bg_{frame_id:03d}", getattr(self, f"combined_xyz_{frame_id:03d}"))
            self._shs_bg = combined_shs
            self._scaling_bg = combined_scales
            self._rotation_bg = combined_rotations
            self._opacity_bg = combined_opacities
            self._t0_bg = combined_t0
            self._duration_bg = combined_duration
            
            print(f"Background gaussians: {num_bg_gaussians}")
            print(f"Foreground gaussians: {num_fg_gaussians}")
        else:
            # Load background data only
            if "means" in background_gs:
                for frame_id in range(49):
                    setattr(self, f"t_xyz_bg_{frame_id:03d}", background_gs['means'].cuda())
            else:
                for frame_id in range(49):
                    setattr(self, f"t_xyz_bg_{frame_id:03d}", background_gs[f'frame_{frame_id:03d}']['means'].cuda())
            self._shs_bg = background_gs['sh0'].cuda()
            self._scaling_bg = background_gs['scales'].cuda()
            self._rotation_bg = background_gs['quats'].cuda()
            self._opacity_bg = background_gs['opacities'].cuda()
            
            num_bg_gaussians = background_gs['scales'].shape[0]
            self._t0_bg = torch.ones(num_bg_gaussians, dtype=torch.float32, device="cuda") * 24
            self._duration_bg = torch.full((num_bg_gaussians,), 49.0, dtype=torch.float32, device="cuda")
            
            print(f"Background gaussians: {num_bg_gaussians}")
    # DEBUG
    def load_simulation(self, round_num):
        """
        直接从round_{round_num}.pt加载模拟数据并创建高斯点
        """
        simulation_path = os.path.join(self.work_dir, "stage3_optimization", "generative_simulation", f"round_{round_num}.pt")
        if not os.path.exists(simulation_path):
            raise FileNotFoundError(f"找不到模拟数据文件: {simulation_path}")
        
        entities_data = torch.load(simulation_path)
        self.num_entities = len(entities_data)
        self.entity_is_rigid = [False] * self.num_entities
        
        # 为每个entity创建高斯点
        for entity_idx, entity_data in enumerate(entities_data):
            # 设置每一帧的xyz

            for frame_id, frame_xyz in enumerate(entity_data['all_frames_xyz']):
                setattr(self, f"t_xyz_{entity_idx}_{frame_id:03d}", frame_xyz)
            
            # 设置其他基本属性
            setattr(self, f"opacity_{entity_idx}_000", entity_data['opacities'])
            setattr(self, f"scaling_{entity_idx}_000", entity_data['scaling'])
            setattr(self, f"t_rotation_{entity_idx}_000", entity_data['rotation'])
            setattr(self, f"sh0_{entity_idx}_000", entity_data['shs'])
            setattr(self, f"t0_{entity_idx}_000", entity_data['t0'])
            setattr(self, f"duration_{entity_idx}_000", entity_data['duration'])
            
            # 如果是刚体，设置每一帧的变换参数
            if 'rigid_rotations' in entity_data:
                if not hasattr(self, 'entity_is_rigid'):
                    self.entity_is_rigid = [False] * self.num_entities
                self.entity_is_rigid[entity_idx] = True
                
                for frame_id in range(len(entity_data['rigid_rotations'])):
                    setattr(self, f"rigid_rotation_{entity_idx}_{frame_id:03d}", entity_data['rigid_rotations'][frame_id])
                    setattr(self, f"rigid_translation_{entity_idx}_{frame_id:03d}", entity_data['rigid_translations'][frame_id])
        
        print(f"成功加载round_{round_num}的模拟数据")

    def get_velocity(self):
        """
        calculate the velocity of the gaussian points
        """
        T = self.num_frames
        xyz = self.get_xyz(T-1)
        prev_xyz = self.get_xyz(T-2)
        velocity = (xyz - prev_xyz) / self.dt
        return velocity
        
    def save_optimized_background(self, round_num = 1):
        """
        保存优化后的背景高斯到指定目录
        保存路径为：work_dir/optimized_background/background.pt
        """
        # 创建保存目录
        save_dir = os.path.join(self.work_dir, "optimized_background", f"round_{round_num}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存数据：每帧的xyz + 共享的其他参数
        background_data = {'splats': {}}
        
        # 保存每一帧的xyz
        for frame_id in range(self.num_frames):
            frame_data = {
                'means': getattr(self, f"t_xyz_bg_{frame_id:03d}").detach().cpu()
            }
            background_data['splats'][f'frame_{frame_id:03d}'] = frame_data
        
        # 保存共享的参数
        background_data['splats']['sh0'] = self._shs_bg.detach().cpu()
        background_data['splats']['scales'] = self._scaling_bg.detach().cpu()
        background_data['splats']['quats'] = self._rotation_bg.detach().cpu()
        background_data['splats']['opacities'] = self._opacity_bg.detach().cpu()
        background_data['splats']['t0'] = self._t0_bg.detach().cpu()
        background_data['splats']['duration'] = self._duration_bg.detach().cpu()
        
        # 保存数据
        save_path = os.path.join(save_dir, "background.pt")
        torch.save(background_data, save_path)
