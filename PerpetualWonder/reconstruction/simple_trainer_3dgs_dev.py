"""
Secondary reconstruction trainer for Gaussian splatting based on simulation-sampled points.

This module performs secondary reconstruction by refitting Gaussian splats using points 
sampled from physics simulation. It takes the physical parameters from simulation results 
and re-optimizes Gaussian splatting representations to better fit the simulated scenes.
"""

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import yaml
# import asdict
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml

import sys
sys.path.append("gsplat/examples")
# Remove current directory from path
current_dir = os.getcwd()
if current_dir in sys.path:
    sys.path.remove(current_dir)
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)

from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Add YAML configuration file path
    config: Optional[str] = None
    # Add bottom threshold parameter
    bottom_threshold: float = 0.0

    # Port for the viewer server
    port: int = 8088

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 10_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [5_000, 10_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [5_000, 10_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [5_000, 10_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 0
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.5  # Initialize to 0.5 for easier learning
    # Initial scale of GS for each entity
    init_scales: Optional[List[float]] = None
    # Default initial scale of GS (used if init_scales is not provided)
    init_scale: float = 0.01
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Weight for alpha loss outside mask
    alpha_loss_weight: float = 1.0

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 0
    # LR for alpha blending weights
    opacities_lr: float = 5e-2  # Reduced learning rate for more stable learning
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False
    # Whether use sub-gaussian
    sub_gaussian: int = 0
    # List of enable_move flags for each entity
    entity_enable_moves: Optional[List[bool]] = None
    # Rate for adding foreground points to background
    rate: float = 0.0
    

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)


def load_physical_params(work_dir: str, round: int) -> List[torch.Tensor]:
    """Load physical_params for multiple entities as foreground Gaussian points"""
    entity_params = []
    entity_idx = 0
    
    while True:
        entity_dir = os.path.join(work_dir, f"physical_params/round_{round}/entity_{entity_idx}")
        if not os.path.exists(entity_dir):
            break
            
        params_path = os.path.join(entity_dir, "0000.npy")

            
        params = np.load(params_path)
        if os.path.exists(os.path.join(entity_dir, "0000_normals.npy")):
            normals = np.load(os.path.join(entity_dir, "0000_normals.npy"))
            params = np.concatenate([params, normals], axis=0)
        if len(params.shape) == 2:
            params = params[None, ...]
        entity_params.append(torch.from_numpy(params).float())
        entity_idx += 1
    
    return entity_params


def build_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """
    Build rotation matrix from Euler angles
    Args:
        rotation: [3] Euler angles (rx, ry, rz)
    Returns:
        R: [3, 3] Rotation matrix
    """
    cos_rx, sin_rx = torch.cos(rotation[0]), torch.sin(rotation[0])
    cos_ry, sin_ry = torch.cos(rotation[1]), torch.sin(rotation[1])
    cos_rz, sin_rz = torch.cos(rotation[2]), torch.sin(rotation[2])
    
    R = torch.zeros((3, 3), device=rotation.device, dtype=rotation.dtype)
    R[0, 0] = cos_ry * cos_rz
    R[0, 1] = -cos_ry * sin_rz
    R[0, 2] = sin_ry
    R[1, 0] = cos_rx * sin_rz + sin_rx * sin_ry * cos_rz
    R[1, 1] = cos_rx * cos_rz - sin_rx * sin_ry * sin_rz
    R[1, 2] = -sin_rx * cos_ry
    R[2, 0] = sin_rx * sin_rz - cos_rx * sin_ry * cos_rz
    R[2, 1] = sin_rx * cos_rz + cos_rx * sin_ry * sin_rz
    R[2, 2] = cos_rx * cos_ry
    
    return R


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 10,
    init_scale: float = 1.0,
    init_scales: Optional[List[float]] = None,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    bottom_threshold: float = 0.0,
    physical_params: Optional[torch.Tensor] = None,
    sub_gaussian: int = 0,
    subgaussian_size: float = 0.02,
    entity_enable_moves: Optional[List[bool]] = None,

) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    # Initialize params list
    params = []

    if physical_params is not None:
        # Store parameters separately for each entity
        for entity_idx, entity_param in enumerate(physical_params):
            # Save anchor points
            anchor_points = entity_param[0, :, :3] + torch.tensor([0, 0, bottom_threshold])  # [N, 3]
            N = len(anchor_points)
            
            # Initialize offset points
            if sub_gaussian > 0:
                points = torch.randn((N * sub_gaussian, 3))
            else:
                points = torch.randn((N, 3))
            
            # Initialize scales
            entity_scale = subgaussian_size if sub_gaussian > 0 else init_scales[entity_idx]
            scales = torch.log(torch.ones((len(points), 1)) * entity_scale)
            
            # Initialize other parameters
            quats = torch.rand((len(points), 4))
            opacities = torch.logit(torch.full((len(points),), init_opacity))
            rgbs = torch.tensor([0.0, 0.0, 0.0]).repeat(len(points), 1)
            colors = torch.zeros((len(points), (sh_degree + 1) ** 2, 3))
            colors[:, 0, :] = rgb_to_sh(rgbs)

            # Set learning rates
            entity_means_lr = means_lr if sub_gaussian > 0 else 0.0
            entity_scales_lr = scales_lr if sub_gaussian > 0 else 0.0
            
            # Add entity parameters
            params.extend([
                (f"anchor_points_{entity_idx}", torch.nn.Parameter(anchor_points, requires_grad=False), 0.0),
                (f"means_{entity_idx}", torch.nn.Parameter(points), entity_means_lr),
                (f"scales_{entity_idx}", torch.nn.Parameter(scales), entity_scales_lr),
                (f"quats_{entity_idx}", torch.nn.Parameter(quats), quats_lr),
                (f"opacities_{entity_idx}", torch.nn.Parameter(opacities), opacities_lr),
                (f"sh0_{entity_idx}", torch.nn.Parameter(colors[:, :1, :]), sh0_lr),
                (f"shN_{entity_idx}", torch.nn.Parameter(colors[:, 1:, :]), shN_lr),
            ])
            
            # Add transformation parameters
            if entity_enable_moves is not None and entity_enable_moves[entity_idx]:
                # Movable entity: parameters are learnable
                translation = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
                rotation = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
                scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
                params.extend([
                    (f"translation_{entity_idx}", translation, 5e-4),
                    (f"rotation_{entity_idx}", rotation, 5e-4),
                    (f"scale_{entity_idx}", scale, 1e-3),
                ])
            else:
                # Immovable entity: parameters fixed at 0
                translation = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=False)
                rotation = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=False)
                scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                params.extend([
                    (f"translation_{entity_idx}", translation, 0.0),
                    (f"rotation_{entity_idx}", rotation, 0.0),
                    (f"scale_{entity_idx}", scale, 0.0),
                ])
            
            print(f"Processing entity {entity_idx}, len(points): {len(points)}, enable_move: {entity_enable_moves[entity_idx] if entity_enable_moves else False}")

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        # Load physical parameters if available

        work_list = cfg.result_dir.split("/")[:-2]
        work_dir = "/".join(work_list)
        self.physical_params = load_physical_params(work_dir+"/stage2_forwardpass", 1)

        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            init_scales=cfg.init_scales,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            bottom_threshold=cfg.bottom_threshold,
            physical_params=self.physical_params,
            sub_gaussian=cfg.sub_gaussian,
            entity_enable_moves=cfg.entity_enable_moves,
        )
        total_gs = sum(len(self.splats[f"means_{i}"]) for i in range(len(self.physical_params)))
        print("Model initialized. Number of GS:", total_gs)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        # Load the background Gaussian splats
        bg_gs_path = os.path.join(work_dir, "stage1_reconstruction/background/background.pt")
        self.bg_splats = torch.load(bg_gs_path)["splats"]
        
        # Load foreground Gaussian points and add part of them to background based on rate
        fg_gs_path = os.path.join(work_dir, "stage1_reconstruction/foreground/foreground.pt")
        if os.path.exists(fg_gs_path):
            fg_splats = torch.load(fg_gs_path)["splats"]
            
            # Select part of foreground points to add to background based on rate parameter
            rate = self.cfg.rate if hasattr(self.cfg, 'rate') else 0
            if rate != 0:
                # Get z values of foreground points and sort
                fg_means = fg_splats["means"].to(self.device)
                print(f"fg_means: {fg_means.shape}")
                z_values = fg_means[:, 2]
                sorted_z, sorted_indices = torch.sort(z_values)
                
                if rate < 0:
                    rate_ = 1 + rate
                else:
                    rate_ = rate
                
                idx = int(len(sorted_z) * abs(rate_))
                z_threshold = sorted_z[idx].item()

                if rate > 0:
                    # Select points with larger z values to add to background
                    fg_to_bg_mask = z_values > z_threshold
                else:
                    # Select points with smaller z values to add to background
                    fg_to_bg_mask = z_values < z_threshold
                
                # Add selected foreground points to background
                # Process means
                fg_means = fg_splats["means"].to(self.device)[fg_to_bg_mask]
                self.bg_splats["means"] = torch.cat([self.bg_splats["means"], fg_means.cpu()], dim=0)
                
                # Process quats
                fg_quats = fg_splats["quats"].to(self.device)[fg_to_bg_mask]
                self.bg_splats["quats"] = torch.cat([self.bg_splats["quats"], fg_quats.cpu()], dim=0)
                
                # Process scales (requires exp and log conversion)
                fg_scales = torch.exp(fg_splats["scales"].to(self.device))[fg_to_bg_mask]
                fg_scales = torch.log(fg_scales)
                self.bg_splats["scales"] = torch.cat([self.bg_splats["scales"], fg_scales.cpu()], dim=0)
                
                # Process opacities (requires sigmoid and logit conversion)
                fg_opacities = torch.sigmoid(fg_splats["opacities"].to(self.device))[fg_to_bg_mask]
                fg_opacities = torch.logit(fg_opacities)
                self.bg_splats["opacities"] = torch.cat([self.bg_splats["opacities"], fg_opacities.cpu()], dim=0)
                
                # Process sh0 and shN
                fg_sh0 = fg_splats["sh0"].to(self.device)[fg_to_bg_mask]
                self.bg_splats["sh0"] = torch.cat([self.bg_splats["sh0"], fg_sh0.cpu()], dim=0)
                
                fg_shN = fg_splats["shN"].to(self.device)[fg_to_bg_mask]
                self.bg_splats["shN"] = torch.cat([self.bg_splats["shN"], fg_shN.cpu()], dim=0)
                
                print(f"Added {fg_to_bg_mask.sum().item()} foreground points to background")


    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,

        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        sub_gaussian: int = 0,
        physical_params: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # Iterate over all entities and compute rendering parameters for each entity
        all_means = []
        all_scales = []
        all_quats = []
        all_opacities = []
        all_colors = []
        
        for entity_idx in range(len(physical_params)):
            # Get anchor points and offsets
            anchor_points = self.splats[f"anchor_points_{entity_idx}"]
            offsets = self.splats[f"means_{entity_idx}"]
            
            # Compute actual rendering positions
            if sub_gaussian > 0:
                N = len(anchor_points)
                anchor_indices = torch.arange(N, device=offsets.device).repeat_interleave(sub_gaussian)
                corresponding_anchors = anchor_points[anchor_indices]
                means = corresponding_anchors + torch.tanh(offsets) * self.cfg.init_scale
            else:
                means = anchor_points
            
            # Apply transformations
            translation = self.splats[f"translation_{entity_idx}"]
            rotation = self.splats[f"rotation_{entity_idx}"]
            scale_factor = self.splats[f"scale_{entity_idx}"]
            
            means = means + translation
            R = build_rotation_matrix(rotation)
            means = torch.mm(means, R.T)
            
            # Get other parameters
            scales = torch.exp(self.splats[f"scales_{entity_idx}"]) * scale_factor
            scales = scales.repeat(1, 3)
            quats = self.splats[f"quats_{entity_idx}"]
            opacities = torch.sigmoid(self.splats[f"opacities_{entity_idx}"])
            colors = torch.cat([self.splats[f"sh0_{entity_idx}"], self.splats[f"shN_{entity_idx}"]], 1)
            
            all_means.append(means)
            all_scales.append(scales)
            all_quats.append(quats)
            all_opacities.append(opacities)
            all_colors.append(colors)
        
        # Concatenate parameters from all entities
        means = torch.cat(all_means, 0)
        scales = torch.cat(all_scales, 0)
        quats = torch.cat(all_quats, 0)
        opacities = torch.cat(all_opacities, 0)
        colors = torch.cat(all_colors, 0)


        # Add the background Gaussian splats and part of foreground Gaussian splats
        if self.bg_splats is not None:
            # Get background Gaussian points
            bg_means = self.bg_splats["means"].to(means.device)
            bg_quats = self.bg_splats["quats"].to(quats.device)
            bg_scales = torch.exp(self.bg_splats["scales"]).to(scales.device)
            bg_opacities = torch.sigmoid(self.bg_splats["opacities"]).to(opacities.device)
            bg_colors = torch.cat([self.bg_splats["sh0"], self.bg_splats["shN"]], 1).to(colors.device)

            means = torch.cat([means, bg_means], 0)
            quats = torch.cat([quats, bg_quats], 0)
            scales = torch.cat([scales, bg_scales], 0)
            opacities = torch.cat([opacities, bg_opacities], 0)
            colors = torch.cat([colors, bg_colors], 0)

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=False,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size


        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = []
        # Create scheduler for means of each entity
        for entity_idx in range(len(self.physical_params)):
            if f"means_{entity_idx}" in self.optimizers:
                schedulers.append(
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.optimizers[f"means_{entity_idx}"], gamma=0.01 ** (1.0 / max_steps)
                    )
                )
     

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)
        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]

      

            height, width = pixels.shape[1:3]

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                # image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
                sub_gaussian=cfg.sub_gaussian,
                physical_params=self.physical_params,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            
            # Compute alpha constraint loss for regions outside mask
            if masks is not None:
                # Alpha constraint for regions outside mask
                outside_mask = ~masks  # [1, H, W]
                outside_alphas = alphas.squeeze(-1) * outside_mask  # [1, H, W]
                alpha_loss_outside = outside_alphas.mean()   # Force regions outside mask to be transparent
                
                # Alpha constraint for regions inside mask
                inside_mask = masks  # [1, H, W]
                inside_alphas = alphas.squeeze(-1) * inside_mask  # [1, H, W]
                alpha_loss_inside = F.binary_cross_entropy(inside_alphas, inside_mask.float())  # Force regions inside mask to be opaque
                
                # Total alpha loss
                alpha_loss = alpha_loss_outside + alpha_loss_inside
                loss += alpha_loss * cfg.alpha_loss_weight
          
            
            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
          
            if masks is not None:
                desc += f"alpha loss={alpha_loss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)


            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                total_gs = sum(len(self.splats[f"means_{i}"]) for i in range(len(self.physical_params)))
                self.writer.add_scalar("train/num_GS", total_gs, step)
                self.writer.add_scalar("train/mem", mem, step)
             
                if masks is not None:
                    self.writer.add_scalar("train/alpha_loss", alpha_loss.item(), step)
               
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                # First save the original multi-entity format
                splats_raw = {}
                for key in self.splats.keys():
                    splats_raw[key] = self.splats[key].clone() if hasattr(self.splats[key], 'clone') else self.splats[key]
                
                # Apply transformations to means and scales and save
                splats_transformed = {}
                for entity_idx in range(len(self.physical_params)):
                    # Copy all parameters for this entity
                    for key in self.splats.keys():
                        if key.endswith(f"_{entity_idx}"):
                            splats_transformed[key] = self.splats[key].clone() if hasattr(self.splats[key], 'clone') else self.splats[key]
                    
                    # Apply transformations to means
                    anchor_points = self.splats[f"anchor_points_{entity_idx}"]
                    means = anchor_points  # Assume no sub_gaussian
                    
                    translation = self.splats[f"translation_{entity_idx}"]
                    rotation = self.splats[f"rotation_{entity_idx}"]
                    scale_factor = self.splats[f"scale_{entity_idx}"]
                    
                    # Apply transformations
                    means = means + translation
                    R = build_rotation_matrix(rotation)
                    means = torch.mm(means, R.T)
                    
                    # Update means
                    splats_transformed[f"means_{entity_idx}"] = means
                    splats_transformed[f"anchor_points_{entity_idx}"] = means
                    
                    # Apply transformations to scales
                    scales = torch.exp(self.splats[f"scales_{entity_idx}"]) * scale_factor
                    splats_transformed[f"scales_{entity_idx}"] = torch.log(scales)
                
                # Save in two formats
                data = {
                    "step": step,
                    "splats": splats_transformed,  # Original multi-entity format (for continuing training)
                }
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                
                torch.save(
                    data,
                    f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )


            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, alpha, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                sub_gaussian=cfg.sub_gaussian,
                physical_params=self.physical_params,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            white_background = torch.ones_like(colors)  # [1, H, W, 3]
            colors_on_white = colors * alpha + white_background * (1 - alpha)

            pixels = pixels * masks.unsqueeze(-1)
            canvas_list = [pixels, colors_on_white]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            total_gs = sum(len(self.splats[f"means_{i}"]) for i in range(len(self.physical_params)))
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": total_gs,
                }
            )
            if cfg.use_bilateral_grid:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                sub_gaussian=cfg.sub_gaussian,
                physical_params=self.physical_params,
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
                camtoworlds=c2w[None],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
                near_plane=render_tab_state.near_plane,
                far_plane=render_tab_state.far_plane,
                radius_clip=render_tab_state.radius_clip,
                eps2d=render_tab_state.eps2d,
                backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
                / 255.0,
                render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
                rasterize_mode=render_tab_state.rasterize_mode,
                camera_model=render_tab_state.camera_model,
                sub_gaussian=self.cfg.sub_gaussian,
                physical_params=self.physical_params,
        )  # [1, H, W, 3]
        total_gs = sum(len(self.splats[f"means_{i}"]) for i in range(len(self.physical_params)))
        render_tab_state.total_gs_count = total_gs
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    # 添加YAML配置支持
    if hasattr(cfg, 'config') and cfg.config is not None:
        yaml_config = load_yaml_config(cfg.config)
        cfg = update_config_from_yaml(cfg, yaml_config)
    
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

def load_yaml_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def update_config_from_yaml(cfg: Config, yaml_config: Dict) -> Config:
    """Update Config object with YAML configuration"""
    # Update key parameters
    cfg.data_dir = yaml_config["work_dir"]+"/stage1_reconstruction/3d"
    cfg.result_dir = yaml_config["work_dir"]+"/stage1_reconstruction/3d"
    cfg.bottom_threshold = yaml_config["simulator_config"]["lift_height"]
    cfg.sub_gaussian = yaml_config["sub-gaussian"]
    # Get init_scale for each entity
    cfg.init_scales = [entity.get("init_scale", cfg.init_scale) for entity in yaml_config["simulator_config"]["entities"]]
    # Get enable_move state for each entity
    cfg.entity_enable_moves = [entity.get("enable_move", False) for entity in yaml_config["simulator_config"]["entities"]]
    cfg.rate = yaml_config["rate"] if "rate" in yaml_config else 0.0
    return cfg

if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    python simple_trainer_3dgs_dev.py --config path/to/config.yaml

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer_3dgs_dev.py --config path/to/config.yaml --steps_scaler 0.25
    ```
    """
    
    cfg = tyro.cli(Config)

    # Add YAML configuration support
    if cfg.config is not None:
        yaml_config = load_yaml_config(cfg.config)
        cfg = update_config_from_yaml(cfg, yaml_config)

    cfg.adjust_steps(cfg.steps_scaler)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    # Initialize distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    world_rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    main(local_rank, world_rank, world_size, cfg)
