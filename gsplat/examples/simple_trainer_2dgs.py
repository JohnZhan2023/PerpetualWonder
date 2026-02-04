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


SEG_FLAG = True

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
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 1_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 0
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
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

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.05
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.008
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 5000
    # Refine GSs every this steps
    refine_every: int = 100

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

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

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Enable normal consistency loss. (Currently for 2DGS only)
    normal_loss: bool = False
    # Weight for normal loss
    normal_lambda: float = 5e-3
    # Iteration to start normal consistency regulerization
    normal_start_iter: int = 7_000

    # Distortion loss. (experimental)
    dist_loss: bool = False
    # Weight for distortion loss
    dist_lambda: float = 1e-1
    # Iteration to start distortion loss regulerization
    dist_start_iter: int = 3_000

    # Model for splatting.
    model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    # Add YAML configuration file path
    config: Optional[str] = None

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)


def create_splats_with_optimizers(
    parser: Union[Parser, Parser_SVC], 
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer], torch.nn.Module]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    objects_dc = torch.randn((N, 16)) * 0.01

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("objects_dc", torch.nn.Parameter(objects_dc), 1e-3),
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
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers, classifier


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
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        # add the first two in trainset to valset
        # 修复：Dataset对象没有extend方法，需要创建新的数据集
        # 方案：创建一个包含验证集和训练集前两个样本的新数据集
        # from torch.utils.data import ConcatDataset
        # train_subset = torch.utils.data.Subset(self.trainset, range(2))  # 取前两个样本
        # self.valset = ConcatDataset([self.valset, train_subset])
        #-----------------------------------------------
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
        feature_dim = 32 if cfg.app_opt else None
        self.splats, splat_optimizers, self.classifier = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type

        # 分离优化器
        self.optimizers = splat_optimizers
        self.classifier_optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=1e-3 * math.sqrt(cfg.batch_size),
            eps=1e-15 / math.sqrt(cfg.batch_size),
            betas=(1 - cfg.batch_size * (1 - 0.9), 1 - cfg.batch_size * (1 - 0.999)),
        )

        if self.model_type == "2dgs":
            key_for_gradient = "gradient_2dgs"
        else:
            key_for_gradient = "means2d"

        # Densification Strategy
        self.strategy = DefaultStrategy(
            verbose=True,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            # refine_scale2d_stop_iter=4000, # splatfacto behavior
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            absgrad=cfg.absgrad,
            revised_opacity=cfg.revised_opacity,
            key_for_gradient=key_for_gradient,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

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

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
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

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
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

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        objects_dc = self.splats["objects_dc"]  # [N, 16]
        # 使用线性层将16维特征映射到2维(前景/背景)
        objects_dc = objects_dc.view(-1, 16)  # [N, 16]
        objects_dc = self.classifier(objects_dc.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)  # [N, 2]
        # 应用softmax得到概率
        objects_dc_cls = torch.softmax(objects_dc, dim=-1)  # [N, 2]
        # 获取分类结果(0表示背景,1表示前景)
        classification = objects_dc_cls.argmax(dim=-1)  # [N]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"

        if self.model_type == "2dgs":
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
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
        elif self.model_type == "2dgs-inria":
            renders, info = rasterization_2dgs_inria_wrapper(
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
                absgrad=self.cfg.absgrad,
                sparse_grad=self.cfg.sparse_grad,
                **kwargs,
            )
            render_colors, render_alphas = renders
            render_normals = info["normals_rend"]
            normals_from_depth = info["normals_surf"]
            render_distort = info["render_distloss"]
            render_median = render_colors[..., 3]

        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            objects_dc_cls,
            classification,  # 返回分割图而不是原始的classification
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
        
        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
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
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                # depths和depths_gt已经是对齐的深度图 [1, H, W, 1]
                depths = data["depths"].to(device)  # [1, H, W, 1]
                depths_gt = data["depths"].to(device)  # [1, H, W, 1]
                masks = data["mask"].to(device) if "mask" in data else None # [1, H, W]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

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
                objects_dc_cls,
                classification,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB+D",
                distloss=self.cfg.dist_loss,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            
            masks = data["mask"].to(device) if "mask" in data else None
            if SEG_FLAG:
                gt_classification = self.check_points_in_mask(self.splats["means"], camtoworlds, Ks, masks, height, width).long()
                cls_loss = F.cross_entropy(objects_dc_cls, gt_classification)
                
                # 添加3D邻近一致性损失
                neighbor_loss = self.loss_cls_3d(
                    predictions=objects_dc_cls,  # 使用softmax后的预测
                    k=5,  # 考虑5个最近邻
                    lambda_val=0.5,  # 权重可以调整
                    max_points=100000,
                    sample_size=500
                )
                cls_loss = cls_loss + neighbor_loss
            else:
                cls_loss = 0.0
                if masks is not None:
                    pixels = pixels * masks[..., None]
                    colors = colors * masks[..., None]

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda + cls_loss * 0.1

            # # 添加mask regularization loss
            if masks is not None and not SEG_FLAG:
                mask_reg_loss = self.compute_mask_regularization_loss(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    masks=masks,
                    height=height,
                    width=width
                )
                if "background" in cfg.data_dir:
                    loss = loss + mask_reg_loss * 0.02
                else:
                    loss = loss + mask_reg_loss * 0.1

            if cfg.depth_loss:
                # depths和depths_gt已经是对齐的深度图 [1, H, W, 1]
                depths = depths.squeeze(-1)  # [1, H, W]
                depths_gt = depths_gt.squeeze(-1)  # [1, H, W]
                assert depths.shape == depths_gt.shape, f"depths and depths_gt shape mismatch: {depths.shape} != {depths_gt.shape}"
                assert depths.shape == masks.shape, f"depths and masks shape mismatch: {depths.shape} != {masks.shape}"
                

                # # 只在前景区域计算scale和shift
                depth_masks = torch.ones_like(depths).bool()
                valid_depths = depths[depth_masks]
                valid_depths_gt = depths_gt[depth_masks]


                # 构建最小二乘问题的矩阵
                A = torch.stack([valid_depths, torch.ones_like(valid_depths)], dim=1)  # [N, 2]
                b = valid_depths_gt  # [N]
                
                # 求解最小二乘问题: min ||A[scale, shift]^T - b||^2
                # 使用正规方程: (A^T A) x = A^T b
                ATA = A.T @ A  # [2, 2]
                ATb = A.T @ b  # [2]
                solution = torch.linalg.solve(ATA, ATb)  # [2]
                scale, shift = solution[0], solution[1]

                # 应用scale和shift到预测的深度图
                aligned_depths = depths * scale + shift

                # 在前景区域计算loss
                depthloss = F.l1_loss(aligned_depths[masks], depths_gt[masks])
                loss += depthloss * cfg.depth_lambda

            if cfg.normal_loss:
                if step > cfg.normal_start_iter:
                    curr_normal_lambda = cfg.normal_lambda
                else:
                    curr_normal_lambda = 0.0
                # normal consistency loss
                normals = normals.squeeze(0).permute((2, 0, 1))
                normals_from_depth *= alphas.squeeze(0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
                normalloss = curr_normal_lambda * normal_error.mean()
                loss += normalloss

            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                distloss = render_distort.mean()
                loss += distloss * curr_dist_lambda

            loss.backward()
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if masks is not None and not SEG_FLAG:
                desc += f"mask_reg={mask_reg_loss.item():.6f}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.dist_loss:
                desc += f"dist loss={distloss.item():.6f}"
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.normal_loss:
                    self.writer.add_scalar("train/normalloss", normalloss.item(), step)
                if cfg.dist_loss:
                    self.writer.add_scalar("train/distloss", distloss.item(), step)
                if cfg.tb_save_image:
                    canvas = (
                        torch.cat([pixels, colors[..., :3]], dim=2)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # 更新classifier
            if SEG_FLAG:
                self.classifier_optimizer.step()
                self.classifier_optimizer.zero_grad(set_to_none=True)
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
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                        "classifier": self.classifier.state_dict(),  # 保存classifier的状态
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                ) 

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

            # 每20步保存一帧
            # if step % 20 == 0:
            #     # 获取当前相机参数
            #     cam_data = all_cameras[camera_idx % len(all_cameras)]
            #     K = cam_data['K'].to(device)
            #     camtoworld = cam_data['camtoworld'].to(device)
            #     height, width = cam_data['height'], cam_data['width']
            #     camera_idx += 1
                
            #     # 渲染图像
            #     with torch.no_grad():
            #         renders, alphas, _, _, _, _, _, _, _ = self.rasterize_splats(
            #             camtoworlds=camtoworld.unsqueeze(0),
            #             Ks=K.unsqueeze(0),
            #             width=width,
            #             height=height,
            #             sh_degree=cfg.sh_degree,
            #             near_plane=cfg.near_plane,
            #             far_plane=cfg.far_plane,
            #         )
            #         frame = renders[0, ..., :3].cpu().numpy()  # [H, W, 3]
            #         frame = (frame * 255).astype(np.uint8)
            #         frames.append(frame)
                    
            #         # 保存当前帧
            #         frame_path = f"{video_dir}/frame_{step:06d}.png"
            #         imageio.imwrite(frame_path, frame)

            # 在训练结束时保存视频
            if step == max_steps - 1:
                video_path = f"{video_dir}/optimization.mp4"
                imageio.mimsave(video_path, frames, fps=10)  # 使用较低的帧率以便观察变化
                print(f"\nSaved video to {video_path}")

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
                # 在最后一步，删除所有背景高斯点
            if step == cfg.max_steps - 1 and SEG_FLAG:
                # save the classifier
                torch.save(self.classifier.state_dict(), f"{self.ckpt_dir}/classifier.pt")


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

    def compute_mask_regularization_loss(self, camtoworlds: torch.Tensor, Ks: torch.Tensor, masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """计算mask外高斯球的regularization loss"""
        means = self.splats["means"]  # [N, 3]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        
        # 检查哪些点在mask外
        points_inside = self.check_points_in_mask(means, camtoworlds, Ks, masks, height, width)  # [N]
        points_outside = ~points_inside  # [N]
        
        if not points_outside.any():
            return torch.tensor(0.0, device=self.device)
            
        # 对mask外的高斯球施加惩罚
        scale_loss = scales[points_outside].mean()  # 使用[N]的布尔索引
        opacity_loss = opacities[points_outside].mean()
        position_loss = torch.abs(means[points_outside]).mean()
        
        # 组合loss
        reg_loss = (scale_loss + opacity_loss + position_loss) * 0.1
        
        return reg_loss

    def reduce_outside_mask_opacity(self, camtoworlds: torch.Tensor, Ks: torch.Tensor, masks: torch.Tensor, height: int, width: int):
        """降低mask外高斯球的opacity"""
        with torch.no_grad():
            means = self.splats["means"]  # [N, 3]
            points_inside = self.check_points_in_mask(means, camtoworlds, Ks, masks, height, width)
            points_outside = ~points_inside
            
            if points_outside.any():
                # 将mask外的高斯球的opacity降低到0.01
                target_opacity = torch.logit(torch.tensor(0.01, device=self.device))
                self.splats["opacities"][points_outside] = target_opacity

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
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
                colors,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                objects_dc_cls,
                segmentation,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            colors = colors[..., :3]  # Take RGB channels
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            # write median depths
            render_median = (render_median - render_median.min()) / (
                render_median.max() - render_median.min()
            )
            # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
            render_median = (
                apply_float_colormap(render_median).detach().cpu().squeeze(0).numpy()
            )

            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_median_depth_{step}.png",
                (render_median * 255).astype(np.uint8),
            )

            # write normals
            normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
            normals_output = (normals * 255).astype(np.uint8)
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_normal_{step}.png", normals_output
            )

            # write normals from depth
            normals_from_depth *= alphas.squeeze(0).detach()
            normals_from_depth = (normals_from_depth * 0.5 + 0.5).cpu().numpy()
            normals_from_depth = (normals_from_depth - np.min(normals_from_depth)) / (
                np.max(normals_from_depth) - np.min(normals_from_depth)
            )
            normals_from_depth_output = (normals_from_depth * 255).astype(np.uint8)
            if len(normals_from_depth_output.shape) == 4:
                normals_from_depth_output = normals_from_depth_output.squeeze(0)
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_normals_from_depth_{step}.png",
                normals_from_depth_output,
            )

            # write distortions

            render_dist = render_distort
            dist_max = torch.max(render_dist)
            dist_min = torch.min(render_dist)
            render_dist = (render_dist - dist_min) / (dist_max - dist_min)
            render_dist = (
                apply_float_colormap(render_dist).detach().cpu().squeeze(0).numpy()
            )
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}_distortions_{step}.png",
                (render_dist * 255).astype(np.uint8),
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means'])}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means"]),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

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
            objects_dc_cls,
            classification,
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
            render_mode="RGB+ED",
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
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


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
