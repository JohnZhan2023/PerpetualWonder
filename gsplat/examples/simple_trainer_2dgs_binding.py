import json
import math
import os
import time
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Optional, Tuple, Union
from pathlib import Path

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import cv2
from datasets.colmap import Dataset, Parser, Parser_SVC
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    apply_depth_colormap,
    colormap,
    knn,
    rgb_to_sh,
    set_random_seed,
)
from gsplat_viewer_2dgs import GsplatViewer, GsplatRenderTabState
from gsplat.rendering import rasterization_2dgs, rasterization_2dgs_inria_wrapper
from gsplat.strategy import DefaultStrategy
from nerfview import CameraState, RenderTabState, apply_float_colormap


def dilate_mask(mask: torch.Tensor, kernel_size: int = 5, iterations: int = 1) -> torch.Tensor:
    """
    对mask进行膨胀操作
    
    Args:
        mask: [B, H, W] 或 [H, W] 的torch.Tensor，值为0或1
        kernel_size: 膨胀核的大小，必须是奇数
        iterations: 膨胀迭代次数
    
    Returns:
        膨胀后的mask，与输入形状相同
    """
    # 确保kernel_size是奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 转换为numpy进行处理
    if mask.dim() == 3:
        # [B, H, W] -> [B, H, W]
        mask_np = mask.detach().cpu().numpy()
        is_batch = True
    else:
        # [H, W] -> [1, H, W]
        mask_np = mask.detach().cpu().numpy()[None]
        is_batch = False
    
    # 创建膨胀核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 对每个batch进行膨胀操作
    dilated_masks = []
    for i in range(mask_np.shape[0]):
        # 将0-1范围的mask转换为0-255范围
        mask_single = (mask_np[i] * 255).astype(np.uint8)
        dilated_mask = cv2.dilate(mask_single, kernel, iterations=iterations)
        # 转换回0-1范围
        dilated_mask = (dilated_mask > 127).astype(np.float32)
        dilated_masks.append(dilated_mask)
    
    # 转换回torch.Tensor
    result = torch.from_numpy(np.stack(dilated_masks)).to(mask.device)
    
    if not is_batch:
        result = result.squeeze(0)
    
    return result


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

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

    # Port for the viewer server
    port: int = 8080

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

    # Initialization strategy
    init_type: str = "sfm"
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 200

    # Degree of spherical harmonics
    sh_degree: int = 0
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    # Add YAML configuration file path
    config: Optional[str] = None
    
    # Mask dilation parameters
    mask_dilation_kernel_size: int = 40
    mask_dilation_iterations: int = 1

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)


def load_physical_params(work_dir: str) -> torch.Tensor:
    """加载physical_params作为前景高斯点"""
    params_path = os.path.join(work_dir, "physical_params/0000.npy")
    params = np.load(params_path)
    return torch.from_numpy(params).float()


def create_splats_with_optimizers(
    parser: Union[Parser, Parser_SVC], 
    init_type: str = "sfm",
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 0,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    physical_params: Optional[torch.Tensor] = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer], torch.nn.Module]:
    if physical_params is not None:
        # 使用physical_params作为初始点
        points = physical_params[:, :3] -  torch.tensor([0, 0, 0.1]) # 取前3列作为xyz坐标
        rgbs = torch.zeros((len(points), 3))  # 初始化为黑色
    elif init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    else:
        raise ValueError("Please specify a correct init_type: sfm or physical_params")

    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    objects_dc = torch.randn((N, 16)) * 0.01

    # 添加全局translation和rotation参数
    global_translation = torch.nn.Parameter(torch.zeros(3), requires_grad=True)  # [3]
    global_rotation = torch.nn.Parameter(torch.zeros(3), requires_grad=True)  # [3] (euler angles)

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points, requires_grad=False), 0.0),  # 固定xyz坐标
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("objects_dc", torch.nn.Parameter(objects_dc), 1e-3),
        # 全局变换参数
        ("global_translation", global_translation, 1e-3),
        ("global_rotation", global_rotation, 1e-3),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    classifier = torch.nn.Conv2d(16, 2, kernel_size=1).to(device)
    param_lrs = {name: lr for name, _, lr in params}
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}]
        )
        for name, lr in param_lrs.items() if lr is not None
    }
    return splats, optimizers, classifier

# 将euler角度转换为旋转矩阵
def euler_to_rotation_matrix(euler_angles):
    # 使用ZYX顺序的euler角
    x, y, z = euler_angles[0], euler_angles[1], euler_angles[2]
    
    # 绕X轴旋转
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(x), -torch.sin(x)],
        [0, torch.sin(x), torch.cos(x)]
    ], device=euler_angles.device, dtype=euler_angles.dtype)
    
    # 绕Y轴旋转
    Ry = torch.tensor([
        [torch.cos(y), 0, torch.sin(y)],
        [0, 1, 0],
        [-torch.sin(y), 0, torch.cos(y)]
    ], device=euler_angles.device, dtype=euler_angles.dtype)
    
    # 绕Z轴旋转
    Rz = torch.tensor([
        [torch.cos(z), -torch.sin(z), 0],
        [torch.sin(z), torch.cos(z), 0],
        [0, 0, 1]
    ], device=euler_angles.device, dtype=euler_angles.dtype)
    
    # 组合旋转矩阵 R = Rz * Ry * Rx
    R = torch.matmul(torch.matmul(Rz, Ry), Rx)
    return R
        
class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=False,  # 不加载深度图
        )
        self.valset = Dataset(self.parser, split="val")

        # 加载背景高斯点
        work_list = cfg.result_dir.split("/")[:-1]
        work_dir = "/".join(work_list)
        background_path = os.path.join(work_dir, "background/background.pt")
        if os.path.exists(background_path):
            print(f"Loading background gaussians from {background_path}")
            self.background_splats = torch.load(background_path, map_location=self.device)
        else:
            print(f"Warning: Background gaussians not found at {background_path}")
            self.background_splats = None
   
        # write in the camera parameters
        def save_camera_params(K, camtoworld, img_id, save_dir):
            os.makedirs(save_dir, exist_ok=True)
            if isinstance(K, torch.Tensor):
                K = K.cpu().numpy()
            if isinstance(camtoworld, torch.Tensor):
                camtoworld = camtoworld.cpu().numpy()
            save_path = os.path.join(save_dir, f"camera_{img_id:04d}.npz")
            np.savez(save_path, K=K, camtoworld=camtoworld)

        # 示例调用
        work_list = cfg.result_dir.split("/")[:-1]
        work_dir = "/".join(work_list)
        cam_pose = os.path.join(work_dir, "camera_pose")

        # 修复索引匹配问题：使用正确的相机ID
        for i in range(len(self.trainset)):
            # 获取第i张图像对应的相机ID
            camera_id = self.parser.camera_ids[i]
            # 使用正确的相机ID获取内参
            K = self.parser.Ks_dict[camera_id]
            camtoworld = self.parser.camtoworlds[i]
            save_camera_params(K, camtoworld, img_id=i, save_dir=cam_pose)
        
        # 保存场景尺度信息
        scene_info = {
            'scene_scale': self.parser.scene_scale,
            'global_scale': cfg.global_scale,
            'final_scale': self.parser.scene_scale * 1.1 * cfg.global_scale,
            'camera_ids': self.parser.camera_ids,
            'num_cameras': len(self.trainset)
        }
        scene_info_path = os.path.join(work_dir, "scene_info.json")
        with open(scene_info_path, 'w') as f:
            json.dump(scene_info, f, indent=2)
        #-----------------------------------------------
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale

        # Model
        feature_dim = None  # 移除app_opt相关的feature_dim设置
        self.splats, splat_optimizers, self.classifier = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=None,
            device=self.device,
            physical_params=load_physical_params(work_dir) if cfg.init_type == "sfm" else None,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        # 移除self.model_type = cfg.model_type

        # 分离优化器
        self.optimizers = splat_optimizers

        # 初始化全局变换参数历史记录
        self.global_transform_history = {
            'translation': [],
            'rotation': [],
            'steps': []
        }

        # 移除pose optimization相关代码
        self.pose_optimizers = []
        self.app_optimizers = []
        # Densification Strategy
        key_for_gradient = 'gradient_2dgs'
        self.str=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    # 在rasterize_splats中完全重写渲染逻辑
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        background_splats: dict = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        # 应用全局变换到前景高斯点
        global_translation = self.splats["global_translation"]  # [3]
        global_rotation = self.splats["global_rotation"]  # [3] (euler angles)
        
        # 获取前景高斯点的数量
        N_foreground = means.shape[0]
        
        # 应用全局变换到前景高斯点
        R = euler_to_rotation_matrix(global_rotation)
        transformed_means = torch.matmul(means, R.T) + global_translation.unsqueeze(0)
        
        # 更新前景高斯点的位置
        means = transformed_means

        if background_splats is not None:
            background_splats = background_splats["splats"]
            means = torch.cat([means, background_splats["means"]], 0).to(self.device)
            quats = torch.cat([quats, background_splats["quats"]], 0).to(self.device)
            
            # 对背景高斯点的scales和opacities应用激活函数
            bg_scales = torch.exp(background_splats["scales"])
            bg_opacities = torch.sigmoid(background_splats["opacities"])
            
            scales = torch.cat([scales, bg_scales], 0).to(self.device)
            opacities = torch.cat([opacities, bg_opacities], 0).to(self.device)
            
            background_colors = torch.cat([background_splats["sh0"], background_splats["shN"]], 1).to(self.device)
            colors = torch.cat([colors, background_colors], 0).to(self.device)

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
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
            sparse_grad=self.cfg.sparse_grad,
            **kwargs,
        )

        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        )

    def loss_cls_3d(self, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
        """计算3D点云的邻近一致性损失
        Args:
            predictions: (N, C) 分类预测
            k: 考虑的邻居数量
            lambda_val: 损失权重
            max_points: 最大点数，超过则下采样
            sample_size: 随机采样点数
        Returns:
            损失值
        """
        # 获取高斯点的位置
        means = self.splats["means"]  # [N, 3]
        N = means.size(0)

        # 条件下采样
        if N > max_points:
            indices = torch.randperm(N, device=means.device)[:max_points]
            means = means[indices]
            predictions = predictions[indices]
            N = max_points

        # 随机采样计算损失的点
        indices = torch.randperm(N, device=means.device)[:sample_size]
        sample_means = means[indices]
        sample_preds = predictions[indices]

        # 计算采样点与所有点之间的距离
        dists = torch.cdist(sample_means, means)  # [sample_size, N]
        _, neighbor_indices = dists.topk(k, largest=False)  # 获取最近的k个点

        # 获取邻居的预测
        neighbor_preds = predictions[neighbor_indices]  # [sample_size, k, C]

        # 计算KL散度
        kl = sample_preds.unsqueeze(1) * (
            torch.log(sample_preds.unsqueeze(1) + 1e-10) - 
            torch.log(neighbor_preds + 1e-10)
        )
        loss = kl.sum(dim=-1).mean()

        # 归一化损失到[0, 1]
        num_classes = predictions.size(1)
        normalized_loss = loss / num_classes

        return lambda_val * normalized_loss

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        # 创建视频保存目录
        video_dir = f"{cfg.result_dir}/optimization_video"
        os.makedirs(video_dir, exist_ok=True)

        # 获取所有相机位姿
        all_cameras = []
        for i in range(len(self.trainset)):
            data = self.trainset[i]
            all_cameras.append({
                'K': data['K'],
                'camtoworld': data['camtoworld'],
                'height': data['image'].shape[0],
                'width': data['image'].shape[1]
            })
        
        # 在train函数中移除pose optimization相关代码
        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        # 移除pose optimization相关代码
        # if cfg.pose_opt:
        #     schedulers.append(
        #         torch.optim.lr_scheduler.ExponentialLR(
        #             self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
        #         )
        #     )

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
        
        # 用于存储视频帧
        frames = []
        camera_idx = 0
        
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
            masks_ori    = data["mask"].to(device)  # [1, H, W]
            
            # 对mask进行膨胀操作
            masks = dilate_mask(masks_ori, kernel_size=cfg.mask_dilation_kernel_size, iterations=cfg.mask_dilation_iterations).bool()
            
            height, width = pixels.shape[1:3]

            # 移除pose noise相关代码
            # if cfg.pose_noise:
            #     camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            # 移除pose optimization相关代码
            # if cfg.pose_opt:
            #     camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            (
                renders,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None


            self.strategy.step_pre_backward(
                params=self.splats,
         [1, H, W, 3]
            masked_pixels = pixels * masks.unsqueeze(-1)  # [1, H, W, 3]
            
            # 计算mask区域的loss
            l1loss = F.l1_loss(masked_colors, masked_pixels)
            ssimloss = 1.0 - self.ssim(
                masked_pixels.permute(0, 3, 1, 2), masked_colors.permute(0, 3, 1, 2)
            )
            
            # 计算mask外区域的alpha约束loss
            outside_mask = ~masks  # [1, H, W]
            outside_alphas = alphas.squeeze(-1) * outside_mask  # [1, H, W]
            alpha_loss = outside_alphas.sum() * 10  # 增大权重以强制mask外区域为透明
            
            # 组合所有loss
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda + alpha_loss
            
            # # 可视化mask覆盖效果（简化版）
            if step % 100 == 0:  # 每100步保存一次可视化
                # 获取原始的未mask处理的数据
                original_pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
                original_colors = renders[..., 0:3] if renders.shape[-1] == 4 else renders
                
                self.visualize_mask_overlay_simple(
                    step=step,
                    gt_rgb=original_pixels,
                    rendered_rgb=original_colors,
                    mask=masks,
                    save_dir=f"{cfg.result_dir}/mask_visualization"
                )

            loss.backward()
            
            # 添加全局变换参数的调试信息
            global_trans = self.splats["global_translation"].detach().cpu().numpy()
            global_rot = self.splats["global_rotation"].detach().cpu().numpy()
            
            desc = (f"loss={loss.item():.3f}| l1={l1loss.item():.3f}| "
                   f"ssim={ssimloss.item():.3f}| alpha={alpha_loss.item():.3f}| "
                   f"sh degree={sh_degree_to_use}| "
                   f"trans=[{global_trans[0]:.3f},{global_trans[1]:.3f},{global_trans[2]:.3f}]| "
                   f"rot=[{global_rot[0]:.3f},{global_rot[1]:.3f},{global_rot[2]:.3f}]| ")
            pbar.set_description(desc)

            # 禁用densification策略
            self.strategy.step_post_backward(
                params=self.splats,
                      # self.strategy.step_pre_backward(...)
            # self.strategy.step_post_backward(...)values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/bingding_gsplat_train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                # add the translation and rotation to the means
                means = self.splats["means"].detach().cpu().numpy()
                R = euler_to_rotation_matrix(self.splats["global_rotation"].detach().cpu().numpy())
                t = self.splats["global_translation"].detach().cpu().numpy()
                means = torch.matmul(means, R.T) + t.unsqueeze(0)
                dict_to_save = self.splats.state_dict().copy()
                dict_to_save["means"] = means

                torch.save(
                    {
                        "step": step,
                        "splats": dict_to_save,
                        "classifier": self.classifier.state_dict(),  # 保存classifier的状态
                    },
                    f"{self.ckpt_dir}/bingding_gsplat_ckpt_{step}.pt",
                ) 
            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                # 计算每秒处理的光线数
                num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]  # H*W*3
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                # Update the viewer state
                self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene
                self.viewer.update(step, num_train_rays_per_step)
                # 在最后一步，删除所有背景高斯点

    def check_points_in_mask(self, means: torch.Tensor, camtoworlds: torch.Tensor, Ks: torch.Tensor, masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """检查点是否在mask内
        Args:
            means: [N, 3] tensor of points
            camtoworlds: [B, 4, 4] camera to world matrices
            Ks: [B, 3, 3] intrinsic matrices
            masks: [B, H, W] binary masks
            height: int, image height
            width: int, image width
        Returns:
            [N] boolean tensor, True if point is inside any mask
        """
        B = camtoworlds.shape[0]
        N = means.shape[0]
        
        # 将点转换到每个相机视图中
        viewmats = torch.linalg.inv(camtoworlds)  # [B, 4, 4]
        
        # 扩展维度以便广播
        means = means.unsqueeze(0)  # [1, N, 3]
        viewmats = viewmats.unsqueeze(1)  # [B, 1, 4, 4]
        Ks = Ks.unsqueeze(1)  # [B, 1, 3, 3]
        
        # 将点转换到相机空间
        ones = torch.ones(1, N, 1, device=means.device)
        points_homo = torch.cat([means, ones], dim=-1)  # [1, N, 4]
        points_cam = torch.matmul(viewmats, points_homo.unsqueeze(-1))  # [B, N, 4, 1]
        points_cam = points_cam.squeeze(-1)[..., :3]  # [B, N, 3]
        
        # 投影到图像平面
        points_2d = torch.matmul(Ks, points_cam.transpose(-2, -1))  # [B, N, 3]
        points_2d = points_2d.transpose(-2, -1)  # [B, N, 3]
        points_2d = points_2d / (points_2d[..., 2:3] + 1e-8)  # [B, N, 3]
        
        # 转换到像素坐标
        pixel_coords = points_2d[..., :2]  # [B, N, 2]
        pixel_x = ((pixel_coords[..., 0] / width) * 2 - 1).clamp(-1, 1)
        pixel_y = ((pixel_coords[..., 1] / height) * 2 - 1).clamp(-1, 1)
        grid = torch.stack([pixel_x, pixel_y], dim=-1)  # [B, N, 2]
        
        # 确保masks是[B, 1, H, W]格式
        if len(masks.shape) == 3:
            masks = masks.unsqueeze(1)  # [B, 1, H, W]
        
        # 采样mask值
        mask_values = []
        for b in range(B):
            grid_b = grid[b:b+1].transpose(0, 1)  # [N, 1, 2]
            mask_b = masks[b:b+1]  # [1, 1, H, W]
            mask_value = F.grid_sample(
                mask_b.float(), 
                grid_b, 
                mode='nearest', 
                align_corners=True
            )  # [1, 1, N, 1]
            mask_values.append(mask_value.squeeze())  # [N]
        
        mask_values = torch.stack(mask_values, dim=0)  # [B, N]
        
        # 如果点在任何一个视图的mask内，则认为它是有效的
        points_in_mask = mask_values.bool().any(dim=0)  # [N]
        
        return points_in_mask

    def visualize_mask_overlay_simple(self, step: int, gt_rgb: torch.Tensor, rendered_rgb: torch.Tensor, mask: torch.Tensor, save_dir: str):
        """简化的mask覆盖可视化：mask透明覆盖在原始图和渲染图上"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 确保输入格式正确
        # gt_rgb和rendered_rgb应该是[B, H, W, C]格式，范围0-1
        # mask应该是[B, H, W]格式，True表示前景，False表示背景
        
        # 创建可视化图像
        B, H, W, C = rendered_rgb.shape
        
        # 1. 原始GT图像
        gt_image = (gt_rgb[0].detach() * 255).cpu().numpy().astype(np.uint8)
        
        # 2. 渲染结果
        rendered_image = (rendered_rgb[0].detach() * 255).cpu().numpy().astype(np.uint8)
        
        # 3. 创建透明mask覆盖效果
        mask_array = mask[0].detach().cpu().numpy()
        alpha = 0.1  # 透明度
        
        # GT图像 + 透明mask覆盖
        gt_with_mask = gt_image.copy()
        gt_with_mask[mask_array] = (
            gt_with_mask[mask_array] * (1 - alpha) + 
            np.array([0, 255, 0]) * alpha  # 使用绿色
        ).astype(np.uint8)
        
        # 渲染图像 + 透明mask覆盖
        rendered_with_mask = rendered_image.copy()
        rendered_with_mask[mask_array] = (
            rendered_with_mask[mask_array] * (1 - alpha) + 
            np.array([0, 255, 0]) * alpha  # 使用绿色
        ).astype(np.uint8)
        
        # 创建组合图：GT+mask | 渲染+mask
        combined_image = np.concatenate([gt_with_mask, rendered_with_mask], axis=1)
        
        # 保存结果
        imageio.imwrite(os.path.join(save_dir, f"step_{step:06d}_mask_overlay_comparison.png"), combined_image)


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
        

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0,
            background_splats=self.background_splats,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "depth":
            # normalize depth to [0, 1]
            depth = render_median
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
        elif render_tab_state.render_mode == "normal":
            render_normals = render_normals * 0.5 + 0.5  # normalize to [0, 1]
            renders = render_normals.cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        else:
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        return renders

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("运行评估...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            (
                renders,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                objects_dc_cls,
                classification,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )  # [1, H, W, 3]
            colors = torch.clamp(renders[..., :3], 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            # 保存图像
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/binding_gsplat_val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            # 保存深度图
            render_median = (render_median - render_median.min()) / (
                render_median.max() - render_median.min()
            )
            render_median = (
                apply_float_colormap(render_median).detach().cpu().squeeze(0).numpy()
            )
            imageio.imwrite(
                f"{self.render_dir}/binding_gsplat_val_{i:04d}_median_depth_{step}.png",
                (render_median * 255).astype(np.uint8),
            )

            # 保存法线图
            normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
            normals_output = (normals * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/binding_gsplat_val_{i:04d}_normal_{step}.png", normals_output
            )

            # 计算指标
            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

            # 添加简化的mask可视化
    

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means'])}"
        )
        
        # 保存统计信息
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means"]),
        }
        with open(f"{self.stats_dir}/binding_gsplat_val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
            
        # 保存到tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("运行轨迹渲染...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="渲染轨迹"):
            renders, _, _, surf_normals, _, _, info = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                background_splats=self.background_splats,
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            surf_normals = (surf_normals - surf_normals.min()) / (
                surf_normals.max() - surf_normals.min()
            )

            # 写入图像
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # 保存为视频
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/binding_gsplat_traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"视频已保存到 {video_dir}/binding_gsplat_traj_{step}.mp4")


def load_yaml_config(config_path: str) -> Dict:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def update_config_from_yaml(cfg: Config, yaml_config: Dict) -> Config:
    """用YAML配置更新Config对象"""
    # 获取Config的所有字段
    config_dict = asdict(cfg)
    
    cfg.data_dir = yaml_config["work_dir"]+"/3d"
    cfg.result_dir = yaml_config["work_dir"]+"/3d"
    return cfg

def main(cfg: Config):
    # 添加YAML配置支持
    if hasattr(cfg, 'config') and cfg.config is not None:
        yaml_config = load_yaml_config(cfg.config)
        cfg = update_config_from_yaml(cfg, yaml_config)
    
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        if "classifier" in ckpt:
            runner.classifier.load_state_dict(ckpt["classifier"])
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[5:-5]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _, surf_normals, _, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            surf_normals = (surf_normals - surf_normals.min()) / (
                surf_normals.max() - surf_normals.min()
            )

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
