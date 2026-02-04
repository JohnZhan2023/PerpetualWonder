import time
import copy
import sys
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from kornia.morphology import dilation
from kornia.core import where as kornia_where
from kornia.geometry.linalg import inverse_transformation, transform_points

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage
from PIL import Image
from einops import rearrange
from kornia.geometry import PinholeCamera
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    RasterizationSettings,
    PointLights,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    Textures,
    SoftSilhouetteShader,
    look_at_view_transform,
)
from pytorch3d.renderer.blending import BlendParams, softmax_rgb_blend, hard_rgb_blend
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer.mesh.textures import TexturesVertex, TexturesUV
from pytorch3d.structures import packed_to_list
from pytorch3d.renderer.points.compositor import _add_background_color_to_images
from pytorch3d.structures import Pointclouds, Meshes
from torchvision.transforms import ToTensor, ToPILImage, Resize
from util.midas_utils import dpt_transform, dpt_512_transform
from util.utils import (
    functbl,
    save_depth_map,
    rotate_pytorch3d_camera,
    translate_pytorch3d_camera,
    SimpleLogger,
    soft_stitching,
)
from util.utils import (
    process_foreground,
    rotate_vector,
    lookAt,
    align_depth_midas,
    save_mask_kps,
    kps_from_quants,
    save_sem_map,
    divide_mask,
    heruistic_reset_depth,
    get_RDF_c2w_from_azimuth_elevation,
)
import rembg
import open3d as o3d

from scene import Scene
from scene.gaussian_model import GaussianModel, BasicPointCloud
from arguments import GSParams
from gaussian_renderer import render
from random import randint

from util.segment_utils import refine_disp_with_segments_2, save_sam_anns
from typing import List, Optional, Tuple, Union
from kornia.morphology import erosion
from syncdiffusion.syncdiffusion_model import SyncDiffusion
import os
from utils.loss import l1_loss, ssim
import matplotlib.pyplot as plt
from scipy.ndimage import label
import xatlas
import omegaconf

sys.path.append("triposr_module/")
# from triposr_module.tsr.system import TSR
# from triposr_module.tsr.bake_texture import bake_texture

sys.path.append("instantmesh_module/")
from instantmesh_module.src.utils.train_util import instantiate_from_config
from instantmesh_module.src.utils.camera_util import (
    get_zero123plus_input_cameras,
    spherical_camera_pose,
)
from instantmesh_module.src.utils.mesh_util import save_obj
from huggingface_hub import hf_hub_download
from torchvision.transforms import v2
import trimesh
from pysdf import SDF

sys.path.append("dust3r_module/")
import roma
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images, load_images_pad
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.cloud_opt.commons import signed_log1p
from dust3r.demo import get_reconstructed_scene, get_3D_model_from_scene
from dust3r.viz import auto_cam_size
from dust3r.utils.device import to_numpy
DUST3R_MODEL_NAME = "/svl/u/zzli/projects/wonder_dy/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

sys.path.append("/svl/u/zzli/projects/wonder_dy/stable-point-aware-3d/")
from spar3d.models.mesh import QUAD_REMESH_AVAILABLE, TRIANGLE_REMESH_AVAILABLE
from spar3d.system import SPAR3D


BG_COLOR = (1, 0, 0)


def debug_vis_func(inp, name="tmp"):
    inp = inp.clone()
    if len(inp.shape) == 4:
        inp = inp[0]
    if len(inp.shape) == 3:
        if inp.shape[0] == 1 or inp.shape[0] == 3:
            inp = inp.permute(1, 2, 0)
            if inp.shape[-1] == 3:
                inp = inp[:, :, [2, 1, 0]]

    if inp.max() <= 1.1:
        inp = inp * 255

    inp = inp.detach().cpu().numpy()
    inp = inp.astype(np.uint8)
    cv2.imwrite(f"{name}.png", inp)


class PointsRenderer(torch.nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(
        self,
        point_clouds,
        return_z=False,
        return_bg_mask=False,
        return_fragment_idx=False,
        **kwargs,
    ) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        zbuf = fragments.zbuf.permute(0, 3, 1, 2)
        fragment_idx = fragments.idx.long().permute(0, 3, 1, 2)
        background_mask = fragment_idx[:, 0] < 0  # [B, H, W]
        images = self.compositor(
            fragment_idx,
            zbuf,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        ret = [images]
        if return_z:
            ret.append(fragments.zbuf)
        if return_bg_mask:
            ret.append(background_mask)
        if return_fragment_idx:
            ret.append(fragments.idx.long())
        if len(ret) == 1:
            ret = images
        return ret


class SoftmaxImportanceCompositor(torch.nn.Module):
    """
    Accumulate points using a softmax importance weighted sum.
    """

    def __init__(
        self,
        background_color: Optional[Union[Tuple, List, torch.Tensor]] = None,
        softmax_scale=1.0,
    ) -> None:
        super().__init__()
        self.background_color = background_color
        self.scale = softmax_scale

    def forward(self, fragments, zbuf, ptclds, **kwargs) -> torch.Tensor:
        """
        Composite features within a z-buffer using importance sum. Given a z-buffer
        with corresponding features and weights, these values are accumulated
        according to softmax(1/z * scale) to produce a final image.

        Args:
            fragments: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
                giving the indices of the nearest points at each pixel, sorted in z-order.
                Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
                feature of the kth closest point (along the z-direction) to pixel (y, x) in
                batch element n.
            zbuf: float32 Tensor of shape (N, points_per_pixel, image_size,
                image_size) giving the depth value of each point in the z-buffer.
                Value -1 means no points assigned to the pixel.
            pt_clds: Packed feature tensor of shape (C, P) giving the features of each point
                (can use RGB for example).

        Returns:
            images: Tensor of shape (N, C, image_size, image_size)
                giving the accumulated features at each point.
        """
        background_color = kwargs.get("background_color", self.background_color)

        zbuf_processed = zbuf.clone()
        zbuf_processed[zbuf_processed < 0] = -1e-4
        importance = 1.0 / (zbuf_processed + 1e-6)
        weights = torch.softmax(importance * self.scale, dim=1)

        fragments_flat = fragments.flatten()
        gathered = ptclds[:, fragments_flat]
        gathered_features = gathered.reshape(
            ptclds.shape[0],
            fragments.shape[0],
            fragments.shape[1],
            fragments.shape[2],
            fragments.shape[3],
        )
        images = (weights[None, ...] * gathered_features).sum(dim=2).permute(1, 0, 2, 3)

        # images are of shape (N, C, H, W)
        # check for background color & feature size C (C=4 indicates rgba)
        if background_color is not None:
            return _add_background_color_to_images(fragments, images, background_color)
        return images


from typing import NamedTuple, Sequence, Union


class MyBlendParams(NamedTuple):
    """
    Data class to store blending params with defaults

    Members:
        sigma (float): For SoftmaxPhong, controls the width of the sigmoid
            function used to calculate the 2D distance based probability. Determines
            the sharpness of the edges of the shape. Higher => faces have less defined
            edges. For SplatterPhong, this is the standard deviation of the Gaussian
            kernel. Higher => splats have a stronger effect and the rendered image is
            more blurry.
        gamma (float): Controls the scaling of the exponential function used
            to set the opacity of the color.
            Higher => faces are more transparent.
        background_color: RGB values for the background color as a tuple or
            as a tensor of three floats.
    """

    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Union[torch.Tensor, Sequence[float]] = (0.0, 0.0, 0.0)


class HardShader(nn.Module):
    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = (
            blend_params if blend_params is not None else MyBlendParams()
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"
            raise ValueError(msg)
        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        # images = softmax_rgb_blend(texels, fragments, blend_params)
        images = hard_rgb_blend(texels, fragments, blend_params)

        return images


class FrameSyn(torch.nn.Module):
    def __init__(
        self,
        config,
        inpainter_pipeline,
        depth_model,
        normal_estimator=None,
        mvdiffusion=None,
        sam_model=None,
    ):
        """This module implement following tasks that are exactly the same in both keyframe generation and new view generation:
        1. Inpainting
        2. Depth estimation
        3. Add new points to a current point cloud

        But it does not implement:
        1. Camera control
        2. Rendering
        3. Initialize point cloud
        4. Anything else
        """
        super().__init__()

        ####### Set up placeholder attributes #######
        self.inpainting_prompt = None
        self.adaptive_negative_prompt = None
        self.current_pc = None
        self.current_pc_sky = None
        self.current_pc_layer = None
        self.current_pc_latest = None  # Only store the valid newly added points for the latest generated scene
        self.current_pc_layer_latest = None  # Only store the valid newly added points for the latest generated scene
        self.current_visible_pc = None
        self.current_visible_pc_init = None
        self.inpainting_resolution = None
        self.border_mask = None
        self.border_size = None
        self.border_image = None
        self.run_dir = None

        self.object_pc_layers = dict()
        self.object_gaussians = dict()
        self.object_gaussians_faces = dict()
        self.object_gaussians_meshes = dict()
        self.object_gaussians_meshes_translation = dict()

        ####### Set up archives #######
        self.image_latest = torch.zeros(1, 3, 512, 512)
        self.sky_mask_latest = torch.zeros(1, 1, 512, 512)
        self.mask_latest = torch.zeros(1, 1, 512, 512)
        self.inpaint_input_image_latest = ToPILImage()(torch.zeros(3, 512, 512))
        self.depth_latest = torch.zeros(1, 1, 512, 512)
        self.disparity_latest = torch.zeros(1, 1, 512, 512)
        self.post_mask_latest = torch.zeros(1, 1, 512, 512)
        self.mask_disocclusion = torch.zeros(1, 1, 512, 512)

        self.kf_idx = 0
        self.images = []
        self.images_layer = []
        self.inpaint_input_images = []
        self.disparities = []
        self.depths = []
        self.masks = []
        self.post_masks = []
        self.cameras = []
        self.cameras_archive = []

        ####### Set up attributes #######
        self.config = config
        self.device = config["device"]

        self.inpainting_pipeline = inpainter_pipeline
        self.use_noprompt = False
        self.negative_inpainting_prompt = config["negative_inpainting_prompt"]
        self.is_upper_mask_aggressive = False
        self.preservation_weight = config["preservation_weight"]
        self.init_focal_length = config["init_focal_length"]

        self.decoder_learning_rate = config["decoder_learning_rate"]
        self.dilate_mask_decoder_ft = config["dilate_mask_decoder_ft"]

        self.depth_model = depth_model
        self.normal_estimator = normal_estimator
        self.depth_model_name = config["depth_model"].lower()
        self.depth_shift = config["depth_shift"]
        self.very_far_depth = config["sky_hard_depth"] * 2

        self.objects_y_min = None

        # 2D pixel points to be used for unprojection and adding new points to PC
        x = torch.arange(512).float() + 0.5
        y = torch.arange(512).float() + 0.5
        self.points = torch.stack(torch.meshgrid(x, y, indexing="ij"), -1)
        self.points = rearrange(self.points, "h w c -> (h w) c").to(self.device)

        self.points_3d_list = []
        self.colors_list = []
        self.floating_point_mask = None
        self.floating_point_mask_list = []
        self.sky_mask_list = []
        self.depth_cache = []
        self.floater_cluster_mask = torch.zeros(1, 1, 512, 512)

        self.mvdiffusion = mvdiffusion
        self.mvdiffusion_steps = 75
        self.sam_model = sam_model

        # self.imgto3d = TSR.from_pretrained(
        #     'stabilityai/TripoSR',
        #     config_name="config.yaml",
        #     weight_name="model.ckpt",
        # )
        # self.imgto3d.renderer.set_chunk_size(8192)
        # self.imgto3d.to(self.device)

        # InstantMesh
        instantmesh_config = OmegaConf.load(
            "instantmesh_module/configs/instant-mesh-large.yaml"
        )
        instantmesh_model_config = instantmesh_config.model_config
        self.instantmesh_infer_config = instantmesh_config.infer_config
        instantmesh_model_ckpt_path = hf_hub_download(
            repo_id="TencentARC/InstantMesh",
            filename="instant_mesh_large.ckpt",
            repo_type="model",
        )
        if 'imgto3d_resolution' in self.config:
            instantmesh_config.model_config.params.grid_res = self.config['imgto3d_resolution']
            
        self.instantmesh = instantiate_from_config(instantmesh_model_config)
        state_dict = torch.load(instantmesh_model_ckpt_path, map_location="cpu")[
            "state_dict"
        ]
        state_dict = {
            k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")
        }
        self.instantmesh.load_state_dict(state_dict, strict=False)
        self.instantmesh.to(self.device)
        # self.instantmesh.init_flexicubes_geometry(self.device, fovy=30.0)
        self.instantmesh.init_flexicubes_geometry(
            self.device,
            fovy=np.rad2deg(2 * np.arctan(512 / (2 * self.init_focal_length))),
        )
        self.instantmesh.eval()
        self.instantmesh_azimuth = np.array([30, 90, 150, 210, 270, 330]).astype(float)
        self.instantmesh_elevation = np.array([20, -10, 20, -10, 20, -10]).astype(float)
        self.instantmesh_radius = 4.0 * 1.0

        self.dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(DUST3R_MODEL_NAME).to(self.device)

        # SPAR3D
        self.spar3d = SPAR3D.from_pretrained(
            "stabilityai/stable-point-aware-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=False,
        )
        self.spar3d.to(self.device)
        self.spar3d.eval()
        self.spar3d_render_azimuth = np.array([30, 90, 150, 210, 270, 330]).astype(float)
        self.spar3d_render_elevation = np.array([20, -10, 20, -10, 20, -10]).astype(float)
        self.spar3d_render_radius = 3.0 # 4.0 * 1.0

        print("image to 3D model initialized ready")

    @torch.no_grad()
    def set_frame_param(
        self, inpainting_resolution, inpainting_prompt, adaptive_negative_prompt
    ):
        self.inpainting_resolution = inpainting_resolution
        self.inpainting_prompt = inpainting_prompt
        self.adaptive_negative_prompt = adaptive_negative_prompt

        # Create mask for inpainting of the right size, white area around the image in the middle
        self.border_mask = torch.ones(
            (1, 1, inpainting_resolution, inpainting_resolution)
        ).to(self.device)
        self.border_size = (inpainting_resolution - 512) // 2
        self.border_mask[
            :,
            :,
            self.border_size : self.inpainting_resolution - self.border_size,
            self.border_size : self.inpainting_resolution - self.border_size,
        ] = 0
        self.border_image = torch.zeros(
            1, 3, inpainting_resolution, inpainting_resolution
        ).to(self.device)

    @torch.no_grad()
    def get_multiview(self, input_image, resolution=512):
        # input_image: [B, 3, h, w]
        rembg_session = rembg.new_session()
        all_output_images = []
        all_output_masks = []
        for id in range(input_image.shape[0]):
            image = input_image[id]
            image = image.permute(1, 2, 0)
            image_np = image.cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            output_image = self.mvdiffusion(
                image_pil,
                num_inference_steps=self.mvdiffusion_steps,
            ).images[0]
            output_images = np.asarray(output_image, dtype=np.float32) / 255.0
            output_images = (
                torch.from_numpy(output_images).permute(2, 0, 1).contiguous().float()
            )  # (3, 960, 640)
            output_images = rearrange(
                output_images, "c (n h) (m w) -> (n m) c h w", n=3, m=2
            )  # (6, 3, 320, 320)

            output_rgbs = []
            output_masks = []
            # use sam or rembg to crop the object
            for bid in range(output_images.shape[0]):
                # image_raw = output_images[bid].permute(1, 2, 0).numpy() * 255
                # image_raw = image_raw.astype(np.uint8)
                # self.sam_model.set_image(image_raw)
                # masks, scores, logits = self.sam_model.predict(point_coords=np.array([[output_images.shape[-2] // 2, output_images.shape[-1] // 2]]))
                # self.sam_model.reset_image()

                image_raw = output_images[bid].permute(1, 2, 0).numpy() * 255
                image_raw = image_raw.astype(np.uint8)
                image_raw = Image.fromarray(image_raw)
                image_raw = rembg.remove(image_raw, session=rembg_session)
                image_raw = np.array(image_raw)

                image_rgb = (
                    torch.from_numpy((image_raw[:, :, :3] / 255).astype(np.float32))
                    .permute(2, 0, 1)
                    .contiguous()
                    .float()
                    .unsqueeze(0)
                )
                image_mask = (
                    torch.from_numpy((image_raw[:, :, 3:] / 255).astype(np.float32))
                    .permute(2, 0, 1)
                    .contiguous()
                    .float()
                    .unsqueeze(0)
                )

                image_rgb = torch.nn.functional.interpolate(
                    image_rgb,
                    (resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                image_mask = torch.nn.functional.interpolate(
                    image_mask,
                    (resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                image_mask = (image_mask > 0.0).float()
                image_rgb = image_rgb * image_mask

                output_rgbs.append(image_rgb)
                output_masks.append(image_mask)

            output_rgbs = torch.cat(output_rgbs, dim=0)
            output_masks = torch.cat(output_masks, dim=0)

            all_output_images.append(output_rgbs)
            all_output_masks.append(output_masks)

        all_output_images = torch.stack(all_output_images, dim=0)
        all_output_masks = torch.stack(all_output_masks, dim=0)
        del rembg_session
        return all_output_images, all_output_masks

    def dust3r_pose_estimation(self, save_path, object_id):
        example_name = self.config['example_name']
        if os.path.exists(os.path.join('./tmp', 'dust3r_pose_estimation', example_name, f'{object_id}.pth')):
            print('loading existing dust3r pose estimation')
            dust3r_pose_estimation = torch.load(os.path.join('./tmp', 'dust3r_pose_estimation', example_name, f'{object_id}.pth'))
            rot_x_deg = dust3r_pose_estimation['rot_x_deg'].to(self.device)
            rot_y_deg = dust3r_pose_estimation['rot_y_deg'].to(self.device)
            rot_z_deg = dust3r_pose_estimation['rot_z_deg'].to(self.device)
            input_view_c2w = dust3r_pose_estimation['input_view_c2w'].to(self.device)
            return rot_x_deg, rot_y_deg, rot_z_deg, input_view_c2w

        os.makedirs(os.path.join(save_path, "for_dust3r"), exist_ok=True)
        
        input_image = Image.open(os.path.join(save_path, "object_image_input.png"))
        input_array = np.array(input_image)
        rgb = input_array[:, :, :3]
        alpha = input_array[:, :, 3]
        white_bg = np.ones_like(rgb) * 255
        alpha_3d = alpha[:, :, np.newaxis] / 255.0
        blended = rgb * alpha_3d + white_bg * (1 - alpha_3d)
        blended = blended.astype(np.uint8)
        blended_pil = Image.fromarray(blended)
        resized = blended_pil.resize((384, 384), Image.Resampling.LANCZOS)
        processed_path = os.path.join(save_path, "for_dust3r", "object_image_0000.png")
        resized.save(processed_path)

        for i in range(6):
            image = Image.open(os.path.join(save_path, f"object_render_image_{i:03d}.png"))
            image = image.resize((384, 384), Image.Resampling.BICUBIC)  # from 320x320 to 384x384, use bicubic interpolation
            image.save(os.path.join(save_path, "for_dust3r", f"object_image_{i+1:04d}.png"))
        
        image_paths = [os.path.join(save_path, "for_dust3r", f"object_image_{i:04d}.png") for i in range(0, 7)]
        images = load_images_pad(image_paths)
        # [6, 4, 4]
        c2ws = get_RDF_c2w_from_azimuth_elevation(self.instantmesh_azimuth, self.instantmesh_elevation, self.instantmesh_radius)
        # Convert azimuth/elevation angles to quaternions
        # Azimuth rotates around y-axis, elevation around x-axis
        # First rotate by elevation around x, then by azimuth around y
        # R = c2ws[:, :3, :3]
        # T = c2ws[:, :3, 3]
        # # Convert to quaternions
        # quats = roma.rotmat_to_unitquat(R)
        # # Create poses with translation
        # poses = torch.zeros(self.instantmesh_azimuth.shape[0], 7, device=self.device)
        # poses[:, :4] = quats
        # poses[:, 4:] = signed_log1p(T)  # Translation in log space
        # pose_msk = [1, 2, 3, 4, 5, 6]

        # For a square image with resolution x and FOV of 50 degrees
        # focal length = (resolution/2) / tan(FOV/2)
        resolution = 384
        fov_rad = torch.deg2rad(torch.tensor(50.0))
        focal = (resolution/2) / torch.tan(fov_rad/2)
        
        # Set both fx and fy to the same focal length since it's a square image
        fx = fy = focal
        
        # Since we pad 64 pixels on both left and right sides (128 total)
        # Need to adjust principal point x coordinate by +64 to account for left padding
        # Original principal point was at 384/2 = 192
        # New principal point should be at 192 + 64 = 256
        intrinsics = torch.tensor(
            [
                [fx, 0, 256],  # Shifted principal point x from 192 to 256
                [0, fy, resolution/2],  # Keep y center at 192
                [0, 0, 1]
            ],
            device=self.device
        )
        intrinsics = intrinsics.repeat(7, 1, 1)

        with torch.enable_grad():
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            output = inference(pairs, self.dust3r_model, self.device, batch_size=1)
            # scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
            # loss = scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)
            
            # scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
            # scene.preset_pose_partial(c2ws, pose_msk=pose_msk)

            scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
            scene.preset_pose([c2ws[i].cpu().numpy() for i in range(0, 6)], [False, True, True, True, True, True, True])
            scene.preset_intrinsics([intrinsics[i].cpu().numpy() for i in range(0, 7)], [True, True, True, True, True, True, True])

            im_poses = to_numpy(scene.get_im_poses())
            vis_cam_size = auto_cam_size(im_poses)
            
            os.makedirs(os.path.join(save_path, "dust3r_mapping_before"), exist_ok=True)
            _ = get_3D_model_from_scene(
                os.path.join(save_path, "dust3r_mapping_before"), silent=False, scene=scene, cam_size=vis_cam_size
            )

            loss = scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

            os.makedirs(os.path.join(save_path, "dust3r_mapping_after"), exist_ok=True)
            _ = get_3D_model_from_scene(
                os.path.join(save_path, "dust3r_mapping_after"), silent=False, scene=scene, cam_size=vis_cam_size
            )
        
        # the output of DUST3R is in RDF convention, x-right, y-down, z-forward
        poses = scene.get_im_poses()
        input_view_pose = poses[0]
        forward_vec = input_view_pose[:3, 2]
        position_vec = input_view_pose[:3, 3]

        use_rotation_matrix = True
        if use_rotation_matrix:
            # Extract rotation angles from camera-to-world rotation matrix
            # For RDF convention (x-right, y-down, z-forward)
            # Using rotation matrix to Euler angles conversion
            rot_mat = input_view_pose[:3,:3]
            
            # Get rotation angles in radians
            # rot_y is rotation around y (pitch)
            rot_y = torch.asin(rot_mat[2,0])
            
            # rot_x is rotation around x (roll)
            rot_x = torch.atan2(-rot_mat[2,1], rot_mat[2,2])
            
            # rot_z is rotation around z (yaw) 
            rot_z = torch.atan2(-rot_mat[1,0], rot_mat[0,0])
            
            # Convert to degrees
            # Adjust rot_y (azimuth) to map negative angles to 180-360 range
            
            rot_x_deg = torch.rad2deg(rot_x)
            rot_y_deg = torch.where(rot_y < 0, 360 + torch.rad2deg(rot_y), torch.rad2deg(rot_y))
            rot_z_deg = torch.rad2deg(rot_z)
        
        else:
            # judge_vec = -forward_vec
            judge_vec = position_vec

            elevation = torch.rad2deg(torch.asin(-judge_vec[1] / judge_vec.norm()))
            azimuth = -torch.rad2deg(torch.atan2(judge_vec[0], judge_vec[2])) + 180

            rot_x_deg = elevation
            rot_y_deg = azimuth
            rot_z_deg = torch.zeros_like(rot_x_deg)
        
        # Transform from RDF (right-down-forward) to LUF (left-up-forward)
        # Need to flip x and y axes
        transform_matrix = torch.tensor([
            [-1, 0, 0, 0],
            [0, -1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], device=input_view_pose.device).float()
        
        # Apply transformation to camera2world matrix
        input_view_c2w = torch.matmul(transform_matrix, input_view_pose)
        input_view_c2w[:, :2] *= -1

        os.makedirs(os.path.join('./tmp', 'dust3r_pose_estimation', example_name), exist_ok=True)
        torch.save({
            'rot_x_deg': rot_x_deg.cpu(),
            'rot_y_deg': rot_y_deg.cpu(),
            'rot_z_deg': rot_z_deg.cpu(),
            'input_view_c2w': input_view_c2w.cpu()
        }, os.path.join('./tmp', 'dust3r_pose_estimation', example_name, f'{object_id}.pth'))
        
        return rot_x_deg, rot_y_deg, rot_z_deg, input_view_c2w

    
    @torch.no_grad()
    def get_spar3d(self, image, save_path):
        # spar3d output directly in pytorch3d space
        mesh, glob_dict = self.spar3d.run_image(
            image,
            bake_resolution=1024,
            remesh="none",
            vertex_count=-1,
            return_points=True,
        )

        (save_path / "spar3d/").mkdir(parents=True, exist_ok=True)
        mesh.export((save_path / "spar3d/mesh.obj").as_posix())

        pt3d_mesh = load_objs_as_meshes([save_path / "spar3d/mesh.obj"], device=self.device)

        def convert_to_textureVertex(textures_uv: TexturesUV, meshes:Meshes) -> TexturesVertex:
            verts_colors_packed = torch.zeros_like(meshes.verts_packed())
            verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()  # (*)
            return TexturesVertex(packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh()))
        
        texture_in_uv = pt3d_mesh.textures
        texture_in_vertex = convert_to_textureVertex(texture_in_uv, pt3d_mesh)
        pt3d_mesh.textures = texture_in_vertex

        azimuths = self.spar3d_render_azimuth
        elevations = self.spar3d_render_elevation
        radius_set = self.spar3d_render_radius

        out_images = []

        for i in range(6):
            radius = radius_set

            K = torch.zeros((1, 4, 4), device=self.device)
            K[0, 0, 0] = self.init_focal_length
            K[0, 1, 1] = self.init_focal_length
            K[0, 0, 2] = 256
            K[0, 1, 2] = 256
            K[0, 2, 3] = 1
            K[0, 3, 2] = 1
            # TODO: further check radius and camera poses here
            R, T = look_at_view_transform(radius, elevations[i], azimuths[i] - 180) 

            camera = PerspectiveCameras(
                K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device
            )

            raster_settings = RasterizationSettings(
                image_size=512, 
                blur_radius=0.0, 
                faces_per_pixel=1, 
            )
            lights = PointLights(device=self.device, location=[[0, 0, 5]])
            
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=camera, 
                    raster_settings=raster_settings
                ),
                # shader=SoftPhongShader(
                #     device=self.device, 
                #     cameras=camera,
                #     lights=lights
                # ),
                shader=HardShader(
                    device=self.device, 
                    cameras=camera,
                )
            )
            rendered_image = renderer(pt3d_mesh)
            rendered_image = rendered_image[0, ..., :3]
            
            rendered_image_np = rendered_image.cpu().numpy()
            rendered_image_np = (rendered_image_np * 255).astype(np.uint8)
            rendered_image_np = Image.fromarray(rendered_image_np)
            rendered_image_np.save(os.path.join(save_path, f"object_render_image_{i:03d}.png"))
            
            rendered_image_torch = rendered_image.permute(2, 0, 1).float().unsqueeze(0)
            out_images.append(rendered_image_torch)
        
        out_images = torch.cat(out_images, dim=0)

        # Export pytorch3d mesh to obj file
        obj_path = os.path.join(save_path, "mesh.obj")
        verts = pt3d_mesh.verts_padded()[0].cpu().numpy()
        faces = pt3d_mesh.faces_padded()[0].cpu().numpy()
        colors = pt3d_mesh.textures.verts_features_padded()[0].cpu().numpy()
        
        tri_mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_colors=colors,
            process=False
        )
        tri_mesh.export(obj_path)

        # spar3d generates too sparse mesh
        if self.config['example_name'] == 'kite_1':
            del tri_mesh
            o3d_mesh = o3d.io.read_triangle_mesh(obj_path)
            densified_tri_mesh = o3d_mesh.subdivide_midpoint(number_of_iterations=1)
            tri_mesh = trimesh.Trimesh(
                vertices=np.asarray(densified_tri_mesh.vertices),
                faces=np.asarray(densified_tri_mesh.triangles),
                vertex_colors=np.asarray(densified_tri_mesh.vertex_colors) * 255,
            )
            os.remove(obj_path)
            tri_mesh.export(obj_path)
        return tri_mesh, out_images
    
    @torch.no_grad()
    def get_instantmesh(self, image, save_path):
        # image: PIL image
        output_image = self.mvdiffusion(
            image,
            num_inference_steps=self.mvdiffusion_steps,
        ).images[0]
        output_images = np.asarray(output_image, dtype=np.float32) / 255.0
        output_images = (
            torch.from_numpy(output_images).permute(2, 0, 1).contiguous().float()
        )  # (3, 960, 640)
        output_images = rearrange(
            output_images, "c (n h) (m w) -> (n m) c h w", n=3, m=2
        )  # (6, 3, 320, 320)

        for id in range(output_images.shape[0]):
            output_image = output_images[id]
            output_image = output_image.permute(1, 2, 0).cpu().numpy()
            output_image = (output_image * 255).astype(np.uint8)
            output_image = Image.fromarray(output_image)
            output_image.save(
                (save_path / f"object_render_image_{id:03d}.png").as_posix()
            )

        input_cameras = get_zero123plus_input_cameras(
            batch_size=1, radius=4.0 * 1.0
        ).to(self.device)

        images = output_images.unsqueeze(0).to(self.device)
        images = v2.functional.resize(
            images, 320, interpolation=3, antialias=True
        ).clamp(0, 1)
        planes = self.instantmesh.forward_planes(images, input_cameras)

        # get mesh
        mesh_path_idx = (save_path / "mesh.obj").as_posix()
        mesh_out = self.instantmesh.extract_mesh(
            planes,
            use_texture_map=False,
            **self.instantmesh_infer_config,
        )
        vertices, faces, vertex_colors = mesh_out
        # from instantmesh code
        vertices = vertices @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        faces = faces[:, [2, 1, 0]]
        vertices = vertices @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        vertices = vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        vertices = vertices @ np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]])
        vertices = vertices @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        final_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
        )

        if self.config['example_name'] == 'jam' and "object_1" in save_path.as_posix():
            # experiment with jam mesh
            # Get y range of vertices
            # TODO: re-implement this with connectivity
            y_coords = final_mesh.vertices[:, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            y_range = y_max - y_min

            # Calculate threshold values
            upper_threshold = y_max - 0.1 * y_range  # Keep vertices below this
            lower_threshold = y_min + 0.1 * y_range  # Keep vertices above this

            # Create mask for vertices to keep
            keep_mask = (y_coords <= upper_threshold) & (y_coords >= lower_threshold)
            
            # Get indices of vertices to keep
            keep_indices = np.where(keep_mask)[0]
            
            # Create a mapping from old vertex indices to new ones
            index_map = np.full(len(final_mesh.vertices), -1)
            index_map[keep_indices] = np.arange(len(keep_indices))
            
            # Get faces that only use kept vertices
            valid_faces_mask = keep_mask[final_mesh.faces].all(axis=1)
            new_faces = index_map[final_mesh.faces[valid_faces_mask]]
            
            # Update mesh with filtered vertices and faces
            final_mesh.vertices = final_mesh.vertices[keep_indices]
            final_mesh.faces = new_faces
            if final_mesh.visual.vertex_colors is not None:
                final_mesh.visual.vertex_colors = final_mesh.visual.vertex_colors[keep_indices]
        
        if self.config['example_name'] == 'food_wine_1' and 'set_new_object' in self.config and self.config['set_new_object']:
            # hack setting for food_wine_1, for glass we first remove its top ceiling
            y_coords = final_mesh.vertices[:, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            y_range = y_max - y_min

            # Calculate threshold values
            upper_threshold = y_max - 0.08 * y_range  # Keep vertices below this
            keep_mask = y_coords <= upper_threshold
            
            # Get indices of vertices to keep
            keep_indices = np.where(keep_mask)[0]
            
            # Create a mapping from old vertex indices to new ones
            index_map = np.full(len(final_mesh.vertices), -1)
            index_map[keep_indices] = np.arange(len(keep_indices))
            
            # Get faces that only use kept vertices
            valid_faces_mask = keep_mask[final_mesh.faces].all(axis=1)
            new_faces = index_map[final_mesh.faces[valid_faces_mask]]
            
            # Update mesh with filtered vertices and faces
            final_mesh.vertices = final_mesh.vertices[keep_indices]
            final_mesh.faces = new_faces
            if final_mesh.visual.vertex_colors is not None:
                final_mesh.visual.vertex_colors = final_mesh.visual.vertex_colors[keep_indices]
            # sometimes this will result in mesh with additional vertices, so save and load again to avoid it in further gaussian initialization
            final_mesh.export(mesh_path_idx)
            final_mesh = trimesh.load(mesh_path_idx)

            # also initialize a new object mesh for wine liquid, save it under object_2
            # Create a new object for wine liquid by sampling points in a cylinder
            # Get the bounding box of the glass mesh
            glass_bbox_min = np.min(final_mesh.vertices, axis=0)
            glass_bbox_max = np.max(final_mesh.vertices, axis=0)
            
            # Define cylinder parameters based on glass dimensions
            cylinder_radius = (glass_bbox_max[0] - glass_bbox_min[0]) * 0.25
            cylinder_height = (glass_bbox_max[1] - glass_bbox_min[1]) * 0.2
            cylinder_center = (glass_bbox_min + glass_bbox_max) / 2
            cylinder_bottom = glass_bbox_min[1] + (glass_bbox_max[1] - glass_bbox_min[1]) * 0.65
            cylinder_center[1] = cylinder_bottom + cylinder_height / 2
            
            # Sample points in a grid
            N = 40
            x = np.linspace(cylinder_center[0] - cylinder_radius, cylinder_center[0] + cylinder_radius, N)
            y = np.linspace(cylinder_bottom, cylinder_bottom + cylinder_height, N) 
            z = np.linspace(cylinder_center[2] - cylinder_radius, cylinder_center[2] + cylinder_radius, N)
            
            # Create 3D grid of points
            xx, yy, zz = np.meshgrid(x, y, z)
            points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
            
            # Keep only points within cylinder radius
            center_xz = np.array([cylinder_center[0], cylinder_center[2]])
            points_xz = points[:,[0,2]]
            distances_from_center = np.sum((points_xz - center_xz)**2, axis=1)
            cylinder_mask = distances_from_center < cylinder_radius**2
            
            # Filter points
            points = points[cylinder_mask]
            points = np.array(points)

            self.wine_pts = points
            self.wine_pts_colors = np.tile(np.array([0.7, 0.1, 0.1]), (len(points), 1))
            self.wine_pts = torch.tensor(self.wine_pts).float().to(self.device)
            self.wine_pts_colors = torch.tensor(self.wine_pts_colors).float().to(self.device)
            self.wine_pts_normals = torch.rand_like(self.wine_pts)
        
        if self.config['example_name'] == 'demo_4' and 'set_new_object' in self.config and self.config['set_new_object'] and "object_0" in save_path.as_posix():
            # hack setting for demo_4, for glass we first remove its top ceiling
            # sdf_checker = SDF(final_mesh.vertices, final_mesh.faces)
            
            y_coords = final_mesh.vertices[:, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            y_range = y_max - y_min

            # Calculate threshold values
            upper_threshold = y_max - 0.02 * y_range  # Keep vertices below this
            keep_mask = y_coords <= upper_threshold
            
            # Get indices of vertices to keep
            keep_indices = np.where(keep_mask)[0]
            
            # Create a mapping from old vertex indices to new ones
            index_map = np.full(len(final_mesh.vertices), -1)
            index_map[keep_indices] = np.arange(len(keep_indices))
            
            # Get faces that only use kept vertices
            valid_faces_mask = keep_mask[final_mesh.faces].all(axis=1)
            new_faces = index_map[final_mesh.faces[valid_faces_mask]]
            
            # Update mesh with filtered vertices and faces
            final_mesh.vertices = final_mesh.vertices[keep_indices]
            final_mesh.faces = new_faces
            if final_mesh.visual.vertex_colors is not None:
                final_mesh.visual.vertex_colors = final_mesh.visual.vertex_colors[keep_indices]
            # sometimes this will result in mesh with additional vertices, so save and load again to avoid it in further gaussian initialization
            final_mesh.export(mesh_path_idx)
            final_mesh = trimesh.load(mesh_path_idx)

            # also initialize a new object mesh for wine liquid, save it under object_2
            # Create a new object for wine liquid by sampling points in a cylinder
            # Get the bounding box of the glass mesh
            glass_bbox_min = np.min(final_mesh.vertices, axis=0)
            glass_bbox_max = np.max(final_mesh.vertices, axis=0)
            
            # Define cylinder parameters based on glass dimensions
            cylinder_radius = (glass_bbox_max[0] - glass_bbox_min[0]) * 0.2
            cylinder_height = (glass_bbox_max[1] - glass_bbox_min[1]) * 0.2
            cylinder_center = (glass_bbox_min + glass_bbox_max) / 2
            cylinder_bottom = glass_bbox_min[1] + (glass_bbox_max[1] - glass_bbox_min[1]) * 0.7
            cylinder_center[1] = cylinder_bottom + cylinder_height / 2
            
            # Sample points in a grid, spacings need to be larger than particle_size=0.01 to avoid collapsing
            Nx = int(2 * cylinder_radius / (0.01 * 1.5))
            Ny = int(cylinder_height / (0.01 * 1.5))
            Nz = Nx
            x = np.linspace(cylinder_center[0] - cylinder_radius, cylinder_center[0] + cylinder_radius, Nx)
            y = np.linspace(cylinder_bottom, cylinder_bottom + cylinder_height, Ny) 
            z = np.linspace(cylinder_center[2] - cylinder_radius, cylinder_center[2] + cylinder_radius, Nz)
            
            # Create 3D grid of points
            xx, yy, zz = np.meshgrid(x, y, z)
            points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
            
            # Keep only points within cylinder radius
            center_xz = np.array([cylinder_center[0], cylinder_center[2]])
            points_xz = points[:,[0,2]]
            distances_from_center = np.sum((points_xz - center_xz)**2, axis=1)
            cylinder_mask = distances_from_center < cylinder_radius**2
            
            # Filter points
            points = points[cylinder_mask]
            points = np.array(points)

            self.wine_pts = points
            self.wine_pts_colors = np.tile(np.array([124 / 255, 2 / 255, 25 / 255]), (len(points), 1))
            self.wine_pts = torch.tensor(self.wine_pts).float().to(self.device)
            self.wine_pts_colors = torch.tensor(self.wine_pts_colors).float().to(self.device)
            self.wine_pts_normals = torch.rand_like(self.wine_pts)
        
        if self.config['example_name'] == 'demo_8' and "object_0" in save_path.as_posix():
            # # just remove the top part of the glass
            y_coords = final_mesh.vertices[:, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            y_range = y_max - y_min

            # Calculate threshold values
            upper_threshold = y_max - 0.02 * y_range  # Keep vertices below this
            keep_mask = y_coords <= upper_threshold
            
            # Get indices of vertices to keep
            keep_indices = np.where(keep_mask)[0]
            
            # Create a mapping from old vertex indices to new ones
            index_map = np.full(len(final_mesh.vertices), -1)
            index_map[keep_indices] = np.arange(len(keep_indices))
            
            # Get faces that only use kept vertices
            valid_faces_mask = keep_mask[final_mesh.faces].all(axis=1)
            new_faces = index_map[final_mesh.faces[valid_faces_mask]]
            
            # Update mesh with filtered vertices and faces
            final_mesh.vertices = final_mesh.vertices[keep_indices]
            final_mesh.faces = new_faces
            if final_mesh.visual.vertex_colors is not None:
                final_mesh.visual.vertex_colors = final_mesh.visual.vertex_colors[keep_indices]
            # sometimes this will result in mesh with additional vertices, so save and load again to avoid it in further gaussian initialization
            final_mesh.export(mesh_path_idx)
            final_mesh = trimesh.load(mesh_path_idx)

        final_mesh.export(mesh_path_idx)
        return final_mesh, output_images

    @torch.no_grad()
    def get_normal(self, image):
        """
        args:
            image: [1, 3, 512, 512]
        """
        # Marigold-my-normal
        # normal = self.normal_estimator(
        #     image,
        #     denoising_steps=10,     # optional
        #     ensemble_size=1,       # optional
        #     processing_res=0,     # optional
        #     match_input_res=True,   # optional
        #     batch_size=0,           # optional
        #     color_map=None,   # optional
        #     show_progress_bar=True, # optional
        #     logger=self.logger,
        # )
        # normal = normal[None].to(dtype=torch.float32)
        # ToPILImage()(normal[0]/2+0.5).save("tmp/normal_my.png")

        # Marigold-official-normal
        normal = self.normal_estimator(
            image * 2 - 1,
            num_inference_steps=10,
            processing_res=768,
            output_prediction_format="pt",
        ).to(
            dtype=torch.float32
        )  # [1, 3, H, W], [-1, 1]
        # ToPILImage()(normal[0]/2+0.5).save("tmp/normal_new.png")
        return normal

    def get_depth(
        self,
        image,
        archive_output=False,
        target_depth=None,
        mask_align=None,
        save_depth_to_cache=False,
        mask_farther=None,
        diffusion_steps=30,
        guidance_steps=8,
    ):
        """
        args:
            image: [1, 3, 512, 512]
            archive_output: if True, then save the depth and disparity to self.depth_latest and self.disparity_latest.
            target_depth: if not None, then use this target depth to condition the depth model.
            mask_align: if not None, then use this mask to align the depth map.
            save_depth_to_cache: if True, then save the depth map to cache.
        """
        assert self.depth_model is not None
        if self.depth_model_name == "midas":
            # MiDaS
            disparity = self.depth_model(dpt_transform(image))
            disparity = torch.nn.functional.interpolate(
                disparity.unsqueeze(1),
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            disparity = disparity.clip(1e-6, max=None)
            depth = 1 / disparity
        if self.depth_model_name == "midas_v3.1":
            img_transformed = dpt_512_transform(image)
            disparity = self.depth_model(img_transformed)
            disparity = torch.nn.functional.interpolate(
                disparity.unsqueeze(1),
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            disparity = disparity.clip(1e-6, max=None)
            depth = 1 / disparity
        elif self.depth_model_name == "zoedepth":
            # ZeoDepth
            depth = self.depth_model(image)["metric_depth"]
        elif self.depth_model_name == "marigold":
            # Marigold
            image_input = (image * 255).byte().squeeze().permute(1, 2, 0)
            image_input = Image.fromarray(image_input.cpu().numpy())
            depth = self.depth_model(
                image_input,
                denoising_steps=diffusion_steps,  # optional
                ensemble_size=1,  # optional
                processing_res=0,  # optional
                match_input_res=True,  # optional
                batch_size=0,  # optional
                color_map=None,  # optional
                show_progress_bar=True,  # optional
                depth_conditioning=self.config["finetune_depth_model"],
                target_depth=target_depth,
                mask_align=mask_align,
                mask_farther=mask_farther,
                guidance_steps=guidance_steps,
                logger=self.logger,
            )

            depth = depth[None, None, :].to(dtype=torch.float32)
            depth /= 200

        elif self.depth_model_name == "marigold_e2e":
            image_input = (image * 255).byte().squeeze().permute(1, 2, 0)
            image_input = Image.fromarray(image_input.cpu().numpy())
            depth = self.depth_model(
                image_input,
                denoising_steps=1,  # optional
                ensemble_size=1,  # optional
                processing_res=0,  # 0 for using input image resolution, original default: 768
                match_input_res=True,  # optional
                batch_size=0,  # optional
                color_map=None,  # optional
                show_progress_bar=True,  # optional
                resample_method="bilinear",
                noise="zeros",
                normals=False,
                # from previous marigold
                # depth_conditioning=self.config["finetune_depth_model"],
                # target_depth=target_depth,
                # mask_align=mask_align,
                # mask_farther=mask_farther,
                # guidance_steps=guidance_steps,
                # logger=self.logger,
            )
            depth = torch.from_numpy(depth.depth_np).to(self.device)
            depth = depth[None, None, :].to(dtype=torch.float32)
            depth /= 200

        elif self.depth_model_name == "depth_anything":
            image_input = (image * 255).byte().squeeze().permute(1, 2, 0)
            image_input = Image.fromarray(image_input.cpu().numpy())
            image_input.save((self.run_dir / "depth_anything_input.png").as_posix())
            raw_image = cv2.imread(
                (self.run_dir / "depth_anything_input.png").as_posix()
            )
            depth = self.depth_model.infer_image(raw_image, 512)
            depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float().to(self.device)
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = depth / 200

        depth = depth + self.depth_shift
        disparity = 1 / depth

        if archive_output:
            self.depth_latest = depth
            self.disparity_latest = disparity

        if save_depth_to_cache:
            self.depth_cache.append(depth)

        return depth, disparity

    @torch.no_grad()
    def inpaint(
        self,
        rendered_image,
        inpaint_mask,
        fill_mask=None,
        fill_mode="cv2_telea",
        self_guidance=False,
        style=None,
        inpainting_prompt=None,
        negative_prompt=None,
        mask_strategy=np.min,
        diffusion_steps=50,
    ):
        # set resolution
        if self.inpainting_resolution > 512 and rendered_image.shape[-1] == 512:
            padded_inpainting_mask = self.border_mask.clone()
            padded_inpainting_mask[
                :,
                :,
                self.border_size : self.inpainting_resolution - self.border_size,
                self.border_size : self.inpainting_resolution - self.border_size,
            ] = inpaint_mask
            padded_rendered_image = self.border_image.clone()
            padded_rendered_image[
                :,
                :,
                self.border_size : self.inpainting_resolution - self.border_size,
                self.border_size : self.inpainting_resolution - self.border_size,
            ] = rendered_image
        else:
            padded_inpainting_mask = inpaint_mask
            padded_rendered_image = rendered_image

        # fill in image
        img = (padded_rendered_image[0].cpu().permute([1, 2, 0]).numpy() * 255).astype(
            np.uint8
        )
        fill_mask = padded_inpainting_mask if fill_mask is None else fill_mask
        fill_mask_ = (fill_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        mask = (padded_inpainting_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        img, _ = functbl[fill_mode](img, fill_mask_)

        # process mask original
        mask_block_size = 8
        mask_boundary = mask.shape[0] // 2
        mask_upper = skimage.measure.block_reduce(
            mask[:mask_boundary, :], (mask_block_size, mask_block_size), mask_strategy
        )
        mask_upper = mask_upper.repeat(mask_block_size, axis=0).repeat(
            mask_block_size, axis=1
        )
        mask_lower = skimage.measure.block_reduce(
            mask[mask_boundary:, :], (mask_block_size, mask_block_size), mask_strategy
        )
        mask_lower = mask_lower.repeat(mask_block_size, axis=0).repeat(
            mask_block_size, axis=1
        )
        mask = np.concatenate([mask_upper, mask_lower], axis=0)

        init_image = Image.fromarray(img)
        mask_image = Image.fromarray(mask)

        if inpainting_prompt is not None:
            self.inpainting_prompt = inpainting_prompt
        if negative_prompt is None:
            negative_prompt = (
                self.adaptive_negative_prompt + self.negative_inpainting_prompt
                if self.adaptive_negative_prompt != None
                else self.negative_inpainting_prompt
            )

        inpainted_image = self.inpainting_pipeline(
            prompt="" if self.use_noprompt else self.inpainting_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=diffusion_steps,
            guidance_scale=0 if self.use_noprompt else 7.5,
            height=self.inpainting_resolution,
            width=self.inpainting_resolution,
            self_guidance=self_guidance,
            inpaint_mask=~padded_inpainting_mask.bool(),
            rendered_image=padded_rendered_image,
        ).images[0]

        # [1, 3, 512, 512]
        inpainted_image = (
            (inpainted_image / 2 + 0.5).clamp(0, 1).to(torch.float32)[None]
        )

        post_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() * 255

        self.post_mask_latest = post_mask
        self.inpaint_input_image_latest = init_image
        self.image_latest = inpainted_image

        return {
            "inpainted_image": inpainted_image,
            "padded_inpainting_mask": padded_inpainting_mask,
            "padded_rendered_image": padded_rendered_image,
        }

    @torch.no_grad()
    def get_current_pc(
        self, is_detach=False, get_sky=False, combine=False, get_layer=False
    ):
        # sky + foreground + layer
        if combine:
            if is_detach:
                return {k: v.detach() for k, v in self.get_combined_pc().items()}
            else:
                return self.get_combined_pc()
        # sky
        elif get_sky:
            if is_detach:
                return {k: v.detach() for k, v in self.current_pc_sky.items()}
            else:
                return self.current_pc_sky
        # layer
        elif get_layer:
            if is_detach:
                return {k: v.detach() for k, v in self.current_pc_layer.items()}
            else:
                return self.current_pc_layer
        # foreground
        else:
            if is_detach:
                return {k: v.detach() for k, v in self.current_pc.items()}
            else:
                return self.current_pc

    @torch.no_grad()
    def get_current_pc_latest(self, get_layer=False):
        if get_layer:
            return {k: v.detach() for k, v in self.current_pc_layer_latest.items()}
        else:
            return {k: v.detach() for k, v in self.current_pc_latest.items()}

    @torch.no_grad()
    def update_current_pc(
        self,
        points,
        colors,
        gen_sky=False,
        gen_layer=False,
        normals=None,
        object_id=None,
        object_mask=None,
        ground_mask=None,
    ):
        if gen_sky:
            if self.current_pc_sky is None:
                self.current_pc_sky = {"xyz": points, "rgb": colors}
            else:
                self.current_pc_sky["xyz"] = torch.cat(
                    [self.current_pc_sky["xyz"], points], dim=0
                )
                self.current_pc_sky["rgb"] = torch.cat(
                    [self.current_pc_sky["rgb"], colors], dim=0
                )
        elif gen_layer:
            if object_id is not None:
                self.object_pc_layers.update(
                    {
                        object_id: {
                            "xyz": points,
                            "rgb": colors,
                            "normals": normals,
                            "num": points.shape[0],
                            "mask": object_mask,
                        }
                    }
                )

            if self.current_pc_layer is None:
                self.current_pc_layer = {
                    "xyz": points,
                    "rgb": colors,
                    "normals": normals,
                }
            else:
                self.current_pc_layer["xyz"] = torch.cat(
                    [self.current_pc_layer["xyz"], points], dim=0
                )
                self.current_pc_layer["rgb"] = torch.cat(
                    [self.current_pc_layer["rgb"], colors], dim=0
                )
                self.current_pc_layer["normals"] = torch.cat(
                    [self.current_pc_layer["normals"], normals], dim=0
                )

            if object_id is not None:
                # if with object id, concat all object pts to train them together
                self.current_pc_layer_latest = {
                    "xyz": self.current_pc_layer["xyz"],
                    "rgb": self.current_pc_layer["rgb"],
                    "normals": self.current_pc_layer["normals"],
                }
            else:
                self.current_pc_layer_latest = {
                    "xyz": points,
                    "rgb": colors,
                    "normals": normals,
                }
        else:
            if self.current_pc is None:
                self.current_pc = {"xyz": points, "rgb": colors}
            else:
                self.current_pc["xyz"] = torch.cat(
                    [self.current_pc["xyz"], points], dim=0
                )
                self.current_pc["rgb"] = torch.cat(
                    [self.current_pc["rgb"], colors], dim=0
                )
            self.current_pc_latest = {"xyz": points, "rgb": colors, "normals": normals}
            if ground_mask is not None:
                self.current_pc.update({"ground_mask": ground_mask})
                self.current_pc_latest.update({"ground_mask": ground_mask})

    @torch.no_grad()
    def get_combined_pc(self):
        if self.current_pc_layer is None:
            pc = {
                "xyz": torch.cat(
                    [self.current_pc["xyz"], self.current_pc_sky["xyz"]], dim=0
                ),
                "rgb": torch.cat(
                    [self.current_pc["rgb"], self.current_pc_sky["rgb"]], dim=0
                ),
            }
        else:
            pc = {
                "xyz": torch.cat(
                    [
                        self.current_pc["xyz"],
                        self.current_pc_sky["xyz"],
                        self.current_pc_layer["xyz"],
                    ],
                    dim=0,
                ),
                "rgb": torch.cat(
                    [
                        self.current_pc["rgb"],
                        self.current_pc_sky["rgb"],
                        self.current_pc_layer["rgb"],
                    ],
                    dim=0,
                ),
            }
        return pc

    @torch.no_grad()
    def push_away_inconsistent_points(self, inconsistent_point_index, depth, mask):
        h, w = depth.shape[2:]
        depth = rearrange(depth.clone(), "b c h w -> (w h b) c")
        extract_mask = rearrange(mask, "b c h w -> (w h b) c")[:, 0].bool()
        depth_extracted = depth[extract_mask]
        if inconsistent_point_index.shape[0] > 0:
            assert depth_extracted.shape[0] >= inconsistent_point_index.max() + 1
        depth_extracted[inconsistent_point_index] = self.very_far_depth
        depth[extract_mask] = depth_extracted
        depth = rearrange(depth, "(w h b) c -> b c h w", w=w, h=h)
        return depth

    @torch.no_grad()
    def archive_latest(self, idx=0, vmax=0.006):
        if self.config["gen_layer"]:
            self.images_layer.append(self.image_latest)
            self.images.append(self.image_latest_init)
        else:
            self.images.append(self.image_latest)
        # render_output = self.render(render_sky=True)
        # render_output = self.render(render_ground=True)
        # self.images_ground.append(render_output['rendered_image'])
        self.masks.append(self.mask_latest)
        self.post_masks.append(self.post_mask_latest)
        self.inpaint_input_images.append(self.inpaint_input_image_latest)
        self.depths.append(self.depth_latest)
        self.disparities.append(self.disparity_latest)

        save_root = Path(self.run_dir) / "images"
        save_root.mkdir(exist_ok=True, parents=True)

        # (save_root / "inpaint_input_images").mkdir(exist_ok=True, parents=True)
        # (save_root / "frames").mkdir(exist_ok=True, parents=True)
        # (save_root / "frames_init").mkdir(exist_ok=True, parents=True)
        # (save_root / "sky_frames").mkdir(exist_ok=True, parents=True)
        # (save_root / "final_frames").mkdir(exist_ok=True, parents=True)
        # (save_root / "masks").mkdir(exist_ok=True, parents=True)
        # (save_root / "post_masks").mkdir(exist_ok=True, parents=True)
        # (save_root / "depth").mkdir(exist_ok=True, parents=True)

        # self.inpaint_input_image_latest.save(save_root / "inpaint_input_images" / f"{idx:03d}.png")
        # ToPILImage()(self.image_latest[0]).save(save_root / "frames" / f"{idx:03d}.png")
        # if self.config['gen_layer']:
        #     ToPILImage()(self.image_latest_init[0]).save(save_root / "frames_init" / f"{idx:03d}.png")
        # # ToPILImage()(self.images_ground[-1][0]).save(save_root / "ground_frames" / f"{idx:03d}.png")
        # ToPILImage()(self.mask_latest[0]).save(save_root / "masks" / f"{idx:03d}.png")
        # ToPILImage()(self.post_mask_latest[0]).save(save_root / "post_masks" / f"{idx:03d}.png")
        # save_depth_map(self.depth_latest.clamp(0).cpu().numpy(), save_root / "depth" / f"{idx:03d}.png", vmax=vmax, save_clean=True)

        if idx == 0:
            with open(Path(self.run_dir) / "config.yaml", "w") as f:
                OmegaConf.save(self.config, f)

    @torch.no_grad()
    def increment_kf_idx(self):
        self.kf_idx += 1

    @torch.no_grad()
    def convert_to_3dgs_traindata(
        self, xyz_scale=1.0, remove_threshold=None, use_no_loss_mask=True
    ):
        """
        args:
            xyz_scale: scale the xyz coordinates by this factor (so that the value range is better for 3DGS optimization and web-viewing).
            remove_threshold: Since 3DGS does not optimize very distant points well, we remove points whose distance to scene origin is greater than this threshold.
        """
        train_datas = []
        W, H = 512, 512
        camera_angle_x = 2 * np.arctan(W / (2 * self.init_focal_length))
        current_pc = self.get_current_pc(is_detach=True)
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()

        if remove_threshold is not None:
            remove_threshold_scaled = remove_threshold * xyz_scale
            mask = np.linalg.norm(pcd_points, axis=0) >= remove_threshold_scaled
            pcd_points = pcd_points[:, ~mask]
            pcd_colors = pcd_colors[~mask]

        frames = []

        for i, img in enumerate(self.images):
            image = ToPILImage()(img[0])
            no_loss_mask = self.no_loss_masks[i][0] if use_no_loss_mask else None
            transform_matrix_pt3d = (
                self.cameras[i].get_world_to_view_transform().get_matrix()[0]
            )
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(
                torch.tensor([-1.0, 1, -1, 1], device=self.device)
            )
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {
                "image": image,
                "transform_matrix": transform_matrix,
                "no_loss_mask": no_loss_mask,
            }
            frames.append(frame)
        train_data = {
            "frames": frames,
            "pcd_points": pcd_points,
            "pcd_colors": pcd_colors,
            "camera_angle_x": camera_angle_x,
            "W": W,
            "H": H,
        }
        train_datas.append(train_data)

        # current_pc = self.get_current_pc(is_detach=True, get_sky=True)
        current_pc = self.sky_pc_downsampled
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = pcd_points / np.linalg.norm(pcd_points, axis=1, keepdims=True)
        pcd_normals = pcd_normals.T

        frames = []

        for i, camera in enumerate(self.sky_cameras):
            self.current_camera = camera
            render_output = self.render(render_sky=True)

            if render_output["inpaint_mask"].mean() > 0:
                render_output["rendered_image"] = inpaint_cv2(
                    render_output["rendered_image"], render_output["inpaint_mask"]
                )
            no_loss_mask = render_output["inpaint_mask"][0]

            image = ToPILImage()(render_output["rendered_image"][0])
            save_root = Path(self.run_dir) / "images"
            # image.save(save_root / "sky_frames" / f"{i:03d}.png")

            transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(
                torch.tensor([-1.0, 1, -1, 1], device=self.device)
            )
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {
                "image": image,
                "transform_matrix": transform_matrix,
                "no_loss_mask": no_loss_mask,
            }
            frames.append(frame)
        train_data_sky = {
            "frames": frames,
            "pcd_points": pcd_points,
            "pcd_colors": pcd_colors,
            "pcd_normals": pcd_normals,
            "camera_angle_x": camera_angle_x,
            "W": W,
            "H": H,
        }
        train_datas.append(train_data_sky)

        if self.config["gen_layer"]:
            current_pc = self.get_current_pc(is_detach=True, get_layer=True)
            pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
            pcd_colors = current_pc["rgb"].cpu().numpy()

            frames = []

            for i, img in enumerate(self.images_layer):
                image = ToPILImage()(img[0])
                no_loss_mask = (
                    self.no_loss_masks_layer[i][0] if use_no_loss_mask else None
                )
                transform_matrix_pt3d = (
                    self.cameras[i].get_world_to_view_transform().get_matrix()[0]
                )
                transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
                transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

                transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

                opengl_to_pt3d = torch.diag(
                    torch.tensor([-1.0, 1, -1, 1], device=self.device)
                )
                transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

                transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
                frame = {
                    "image": image,
                    "transform_matrix": transform_matrix,
                    "no_loss_mask": no_loss_mask,
                }
                frames.append(frame)
            train_data_layer = {
                "frames": frames,
                "pcd_points": pcd_points,
                "pcd_colors": pcd_colors,
                "camera_angle_x": camera_angle_x,
                "W": W,
                "H": H,
            }
            train_datas.append(train_data_layer)

        return train_datas

    @torch.no_grad()
    def convert_to_3dgs_traindata_latest(
        self,
        xyz_scale=1.0,
        points_3d=None,
        colors=None,
        use_no_loss_mask=False,
        use_only_latest_frame=True,
    ):
        """
        args:
            xyz_scale: scale the xyz coordinates by this factor (so that the value range is better for 3DGS optimization and web-viewing).
        """
        W, H = 512, 512
        camera_angle_x = 2 * np.arctan(W / (2 * self.init_focal_length))
        current_pc = self.get_current_pc_latest()
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = current_pc["normals"].cpu().numpy()

        frames = []

        images = self.images
        for i, img in enumerate(images):
            if use_only_latest_frame and i != len(images) - 1:
                continue
            image = ToPILImage()(img[0])
            no_loss_mask = self.no_loss_masks[i][0] if use_no_loss_mask else None
            transform_matrix_pt3d = (
                self.cameras_archive[i].get_world_to_view_transform().get_matrix()[0]
            )
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(
                torch.tensor([-1.0, 1, -1, 1], device=self.device)
            )
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {
                "image": image,
                "transform_matrix": transform_matrix,
                "no_loss_mask": no_loss_mask,
            }
            frames.append(frame)
        train_data = {
            "frames": frames,
            "pcd_points": pcd_points,
            "pcd_colors": pcd_colors,
            "pcd_normals": pcd_normals,
            "camera_angle_x": camera_angle_x,
            "W": W,
            "H": H,
        }

        return train_data

    @torch.no_grad()
    def convert_to_3dgs_traindata_latest_layer(
        self, xyz_scale=1.0, points_3d=None, colors=None, use_only_latest_frame=True
    ):
        """
        args:
            xyz_scale: scale the xyz coordinates by this factor (so that the value range is better for 3DGS optimization and web-viewing).
        return:
            train_data: Original image and the point cloud of only occluding objects
            train_data_layer: Base image (original with inpainted regions) and the point cloud of the base layer
        """
        W, H = 512, 512
        camera_angle_x = 2 * np.arctan(W / (2 * self.init_focal_length))

        # if points_3d is None or colors is None:
        #     current_pc = self.get_current_pc(is_detach=True)
        #     pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        #     pcd_colors = current_pc["rgb"].cpu().numpy()
        # else:
        #     pcd_points = points_3d.permute(1, 0).cpu().numpy() * xyz_scale
        #     pcd_colors = colors.cpu().numpy()

        current_pc = self.get_current_pc_latest(get_layer=True)
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = current_pc["normals"].cpu().numpy()
        frames = []
        images = self.images
        for i, img in enumerate(images):
            if use_only_latest_frame and i != len(images) - 1:
                continue
            image = ToPILImage()(img[0])
            transform_matrix_pt3d = (
                self.cameras_archive[i].get_world_to_view_transform().get_matrix()[0]
            )
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(
                torch.tensor([-1.0, 1, -1, 1], device=self.device)
            )
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {
                "image": image,
                "transform_matrix": transform_matrix,
                "no_loss_mask": None,
            }
            frames.append(frame)
        train_data = {
            "frames": frames,
            "pcd_points": pcd_points,
            "pcd_colors": pcd_colors,
            "pcd_normals": pcd_normals,
            "camera_angle_x": camera_angle_x,
            "W": W,
            "H": H,
        }

        current_pc = self.get_current_pc_latest()
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = current_pc["normals"].cpu().numpy()
        frames = []
        images = self.images_layer
        for i, img in enumerate(images):
            if use_only_latest_frame and i != len(images) - 1:
                continue
            image = ToPILImage()(img[0])
            transform_matrix_pt3d = (
                self.cameras_archive[i].get_world_to_view_transform().get_matrix()[0]
            )
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(
                torch.tensor([-1.0, 1, -1, 1], device=self.device)
            )
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {
                "image": image,
                "transform_matrix": transform_matrix,
                "no_loss_mask": None,
            }
            frames.append(frame)
        train_data_layer = {
            "frames": frames,
            "pcd_points": pcd_points,
            "pcd_colors": pcd_colors,
            "pcd_normals": pcd_normals,
            "camera_angle_x": camera_angle_x,
            "W": W,
            "H": H,
        }

        return train_data, train_data_layer

    @torch.no_grad()
    def convert_to_3dgs_traindata_list_latest_layer(
        self, xyz_scale=1.0, points_3d=None, colors=None, use_only_latest_frame=True
    ):
        """
        args:
            xyz_scale: scale the xyz coordinates by this factor (so that the value range is better for 3DGS optimization and web-viewing).
        return:
            train_data: Original image and the point cloud of only occluding objects
            train_data_layer: Base image (original with inpainted regions) and the point cloud of the base layer
        """
        W, H = 512, 512
        camera_angle_x = 2 * np.arctan(W / (2 * self.init_focal_length))

        # if points_3d is None or colors is None:
        #     current_pc = self.get_current_pc(is_detach=True)
        #     pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        #     pcd_colors = current_pc["rgb"].cpu().numpy()
        # else:
        #     pcd_points = points_3d.permute(1, 0).cpu().numpy() * xyz_scale
        #     pcd_colors = colors.cpu().numpy()

        # for multiple objects, we need to get the pc for each object
        train_data_list = []
        object_pcs = self.object_pc_layers
        object_gaussians = self.object_gaussians
        for object_id, object_pc in object_pcs.items():
            pcd_points = object_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
            pcd_colors = object_pc["rgb"].cpu().numpy()
            pcd_normals = object_pc["normals"].cpu().numpy()
            pcd_mask = object_pc["mask"].cpu().numpy()
            frames = []
            images = self.images
            for i, img in enumerate(images):
                if use_only_latest_frame and i != len(images) - 1:
                    continue
                image = ToPILImage()(img[0])
                transform_matrix_pt3d = (
                    self.cameras_archive[i]
                    .get_world_to_view_transform()
                    .get_matrix()[0]
                )
                transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
                transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

                transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

                opengl_to_pt3d = torch.diag(
                    torch.tensor([-1.0, 1, -1, 1], device=self.device)
                )
                transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

                transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
                frame = {
                    "image": image,
                    "transform_matrix": transform_matrix,
                    "no_loss_mask": torch.from_numpy(1 - pcd_mask[0]),
                }
                frames.append(frame)
            train_data = {
                "frames": frames,
                "pcd_points": pcd_points,
                "pcd_colors": pcd_colors,
                "pcd_normals": pcd_normals,
                "camera_angle_x": camera_angle_x,
                "W": W,
                "H": H,
            }
            if object_id in object_gaussians.keys():
                object_gaussian = object_gaussians.pop(object_id)
                train_data["gaussians"] = object_gaussian
                train_data["faces"] = self.object_gaussians_faces[object_id]
                train_data["mesh_path"] = self.object_gaussians_meshes[object_id]
                train_data["mesh_translation"] = self.object_gaussians_meshes_translation[object_id]

            train_data_list.append(train_data)

        current_pc = self.get_current_pc_latest()
        pcd_points = current_pc["xyz"].permute(1, 0).cpu().numpy() * xyz_scale
        pcd_colors = current_pc["rgb"].cpu().numpy()
        pcd_normals = current_pc["normals"].cpu().numpy()
        frames = []
        images = self.images_layer
        for i, img in enumerate(images):
            if use_only_latest_frame and i != len(images) - 1:
                continue
            image = ToPILImage()(img[0])
            transform_matrix_pt3d = (
                self.cameras_archive[i].get_world_to_view_transform().get_matrix()[0]
            )
            transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            transform_matrix_w2c_pt3d[:3, 3] *= xyz_scale

            transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()

            opengl_to_pt3d = torch.diag(
                torch.tensor([-1.0, 1, -1, 1], device=self.device)
            )
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {
                "image": image,
                "transform_matrix": transform_matrix,
                "no_loss_mask": None,
            }
            frames.append(frame)
        
        # for background, take the y_min of objects into consideration
        # if self.objects_y_min is not None:
        #     pcd_points_lowery = pcd_points[1, :] < self.objects_y_min.item()
        #     ground_mask = current_pc["ground_mask"].cpu().numpy()
        #     pcd_points_lowery = pcd_points_lowery & ground_mask
        #     pcd_points[1, pcd_points_lowery] = self.objects_y_min.item()
        #     ground_value = self.objects_y_min.item()
        #     self.objects_y_min = None
        ground_value = 0.000

        train_data_layer = {
            "frames": frames,
            "pcd_points": pcd_points,
            "pcd_colors": pcd_colors,
            "pcd_normals": pcd_normals,
            "camera_angle_x": camera_angle_x,
            "W": W,
            "H": H,
            "ground_value": ground_value,
        }

        return train_data_list, train_data_layer

    @torch.no_grad()
    def get_knn_mask(self, pad_width=1):
        """
        Clean depth map by removing floating points with KNN heuristic over multiple iterations.

        Args:
        - pad_width: Padding width for the depth map processing.

        Returns:
        - mask which indicates floating points after specified iterations.
        """
        print("-- knn heuristic, removing floating points...")
        depth_map = self.depth_latest.squeeze().detach().cpu().numpy()
        height, width = depth_map.shape
        padded_depth_map = np.pad(
            depth_map, pad_width=pad_width, mode="constant", constant_values=0
        )
        cleaned_depth_map = np.zeros_like(depth_map)

        for dy in range(-pad_width, pad_width + 1):
            for dx in range(-pad_width, pad_width + 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor_diff = np.abs(
                    padded_depth_map[
                        pad_width + dy : height + pad_width + dy,
                        pad_width + dx : width + pad_width + dx,
                    ]
                    - depth_map
                )
                cleaned_depth_map += neighbor_diff > 0.00001

        knn_mask = torch.from_numpy(cleaned_depth_map == 8)
        print("-- floating points ratio: {}".format(knn_mask.float().mean()))
        ToPILImage()(knn_mask.float()).save(
            self.run_dir / "images" / "knn_masks" / f"{self.kf_idx:02d}_knn_mask.png"
        )

        return knn_mask

    @torch.no_grad()
    def update_current_pc_by_kf(
        self,
        valid_mask=None,
        gen_layer=False,
        image=None,
        depth=None,
        camera=None,
        multiview=False,
        imgto3D=False,
        object_id=None,
        ground_mask=None,
    ):
        """
        Use self.image_latest and self.depth_latest to update current_pc.
        args:
            valid_mask: if None, then use inpaint_mask (given by rendered_depth == 0) to extract new points.
                        if not None, should be [B, C, H, W], then just valid_mask to extract new points.
        """
        if image is None:
            image = self.image_latest
        if depth is None:
            depth = self.depth_latest
        if camera is None:
            camera = self.current_camera
        kf_camera = convert_pytorch3d_kornia(camera, self.init_focal_length)
        point_depth = rearrange(depth, "b c h w -> (w h b) c")
        normals = self.get_normal(image[0])
        normals[
            :, 1:
        ] *= -1  # Marigold normal is opengl format; make it opencv format here

        normals_world = kf_camera.rotation_matrix.inverse() @ rearrange(
            normals, "b c h w -> b c (h w)"
        )

        # if not gen_layer:
        #     normal_viz = (normals + 1) / 2
        #     ToPILImage()(normal_viz[0]).save('tmp/screen_pred_normal.png')
        #     normal_viz = rearrange(normals_world, 'b c (h w) -> b c h w', h=512) / 2 + 0.5
        #     ToPILImage()(normal_viz[0]).save('tmp/screen_pred_normal_world.png')

        #     ToPILImage()(self.image_latest[0]).save('tmp/image_latest.png')

        normals = rearrange(normals_world, "b c (h w) -> b c h w", h=512)
        new_normals = rearrange(normals, "b c h w -> (w h b) c")
        new_points_3d = kf_camera.unproject(self.points, point_depth)

        image_points_3d = new_points_3d
        image_points_3d = rearrange(
            image_points_3d, "(w h b) c -> b c h w", h=512, w=512, b=1
        )
        image_points_3d = image_points_3d * valid_mask

        new_colors = rearrange(image, "b c h w -> (w h b) c")

        if valid_mask is not None:
            extract_mask = rearrange(valid_mask, "b c h w -> (w h b) c")[:, 0].bool()
            new_points_3d = new_points_3d[extract_mask]
            new_colors = new_colors[extract_mask]
            new_normals = new_normals[extract_mask]

            if ground_mask is not None:
                ground_mask = rearrange(ground_mask, "b c h w -> (w h b) c")[:, 0].bool()
                ground_mask = ground_mask[extract_mask]

        if multiview and valid_mask is not None:
            mvd_path = self.run_dir / "images" / "multiview_diffusion"
            foreground_image, foreground_mask, update_intrinsics_parameters = (
                process_foreground(image, valid_mask, res=512)
            )
            foreground_depth, _, _ = process_foreground(depth, valid_mask, res=512)

            valid_depth = point_depth[extract_mask]
            valid_depth_max, valid_depth_min, valid_depth_mean = (
                valid_depth.max(),
                valid_depth.min(),
                valid_depth.mean(),
            )

            center_point_cano = torch.tensor(
                [
                    0.5 * (new_points_3d[:, 0].max() - new_points_3d[:, 0].min())
                    + new_points_3d[:, 0].min(),
                    0.5 * (new_points_3d[:, 1].max() - new_points_3d[:, 1].min())
                    + new_points_3d[:, 1].min(),
                    0.5 * (new_points_3d[:, 2].max() - new_points_3d[:, 2].min())
                    + new_points_3d[:, 2].min(),
                ]
            ).to(self.device)
            # heuristic
            center_point_cano = (
                center_point_cano - kf_camera.translation_vector.reshape(-1)
            ) * 0.6 + kf_camera.translation_vector.reshape(-1)
            cano_vector = center_point_cano - kf_camera.translation_vector.reshape(-1)

            mvd_camera = convert_pytorch3d_kornia(
                camera,
                self.init_focal_length,
                update_intrinsics_parameters=update_intrinsics_parameters,
                new_size=foreground_image.shape[-1],
            )
            mvd_intrinsics = mvd_camera.intrinsics
            original_intrinsics = kf_camera.intrinsics
            mvd_images, mvd_masks = self.get_multiview(
                foreground_image, resolution=foreground_image.shape[-1]
            )
            # [6, 3, 320, 320]
            # relative elevation in [30, 90, 150, 210, 270, 330]
            # relative azimuth in [20, -10, 20, -10, 20, -10]
            mvd_images = mvd_images[0]
            mvd_masks = mvd_masks[0]
            view_azimuths = torch.Tensor([30, 90, 150, 210, 270, 330]).to(
                device=self.device
            )
            view_elevations = torch.Tensor([20, -10, 20, -10, 20, -10]).to(
                device=self.device
            )
            use_view_ids = [0, 1, 2, 3, 4, 5]

            mvd_images = mvd_images[use_view_ids]
            mvd_masks = mvd_masks[use_view_ids]
            view_elevations = view_elevations[use_view_ids]
            view_azimuths = view_azimuths[use_view_ids]

            for vid in range(len(use_view_ids)):
                mvd_image_np = (
                    mvd_images[vid].permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)
                mvd_mask_np = (
                    mvd_masks[vid].permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)

                view_mask_np = mvd_masks[vid].permute(1, 2, 0).cpu().numpy()
                kernel_size = 10
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                view_mask_np_erode = cv2.erode(view_mask_np, kernel, iterations=1)
                view_mask_np = view_mask_np_erode

                cv2.imwrite(
                    (mvd_path / f"view{vid}_image.png").as_posix(),
                    mvd_image_np[:, :, [2, 1, 0]],
                )
                cv2.imwrite(
                    (mvd_path / f"view{vid}_mask_original.png").as_posix(), mvd_mask_np
                )
                cv2.imwrite(
                    (mvd_path / f"view{vid}_mask.png").as_posix(),
                    (view_mask_np * 255).astype(np.uint8),
                )

                view_mask = (
                    torch.Tensor(view_mask_np).to(self.device).unsqueeze(0).unsqueeze(0)
                )

                view_vector = rotate_vector(
                    cano_vector, view_elevations[vid], view_azimuths[vid]
                )
                new_camera_pos = center_point_cano - view_vector
                new_camera_c2w, new_camera_w2c = lookAt(
                    new_camera_pos,
                    center_point_cano,
                    torch.Tensor([0, 1, 0]).to(self.device),
                )

                pt3d_to_kornia = torch.diag(
                    torch.tensor([-1.0, -1, 1, 1], device=self.device)
                )
                new_camera_w2c_kornia = pt3d_to_kornia @ new_camera_w2c
                new_camera_kornia = PinholeCamera(
                    mvd_intrinsics,
                    # original_intrinsics,
                    new_camera_w2c_kornia,
                    torch.tensor([foreground_image.shape[-2]], device="cuda"),
                    torch.tensor([foreground_image.shape[-2]], device="cuda"),
                )

                # get image depth and normal
                view_normals = self.get_normal(mvd_images[vid])
                view_normals[
                    :, 1:
                ] *= -1  # Marigold normal is opengl format; make it opencv format here
                view_normals_world = (
                    new_camera_kornia.rotation_matrix.inverse()
                    @ rearrange(view_normals, "b c h w -> b c (h w)")
                )
                view_normals = rearrange(
                    view_normals_world,
                    "b c (h w) -> b c h w",
                    h=foreground_image.shape[-1],
                )
                view_normals = rearrange(view_normals, "b c h w -> (w h b) c")

                with torch.no_grad():
                    # exist_points_3d = new_points_3d.clone()
                    # new_camera_project_matrix = new_camera_kornia.intrinsics @ new_camera_kornia.extrinsics
                    # exist_points_2d = transform_points(new_camera_project_matrix, exist_points_3d)
                    # z_vec = exist_points_2d[..., -1:]
                    # z_vec_mask = torch.abs(z_vec) > 1e-8
                    # z_scale = kornia_where(z_vec_mask, 1.0 / (z_vec + 1e-8), torch.ones_like(z_vec))
                    # exist_points_2d = z_scale * exist_points_2d[..., :-1]

                    # view_valid_depth_mask = (exist_points_2d[:,0] <= 511.5) & (exist_points_2d[:,1] <= 511.5) & (exist_points_2d[:,0] > 0.5) & (exist_points_2d[:,1] > 0.5)
                    # exist_points_2d = exist_points_2d[view_valid_depth_mask]
                    # z_vec = z_vec[view_valid_depth_mask]

                    # view_target_depth = torch.zeros_like(depth)[0,0]
                    # view_target_depth_mask = torch.zeros_like(depth)[0,0]
                    # view_target_depth_index = (exist_points_2d - 0.5).long()    # in [w,h]
                    # view_target_depth[view_target_depth_index[:,1], view_target_depth_index[:,0]] = z_vec.squeeze(-1)
                    # view_target_depth_mask[view_target_depth_index[:,1], view_target_depth_index[:,0]] = 1

                    # view_target_depth_mask = view_target_depth_mask.unsqueeze(0).unsqueeze(0)
                    # view_target_depth = view_target_depth.unsqueeze(0).unsqueeze(0)
                    # view_target_depth_vis = (view_target_depth - view_target_depth.min()) / (view_target_depth.max() - view_target_depth.min())
                    # view_target_depth_vis = (view_target_depth_vis.cpu().numpy() * 255).astype(np.uint8)
                    # cv2.imwrite((mvd_path / f'view{vid}_depth_project_before.png').as_posix(), view_target_depth_vis[0,0])
                    # view_target_depth_mask = view_target_depth_mask.long()

                    view_target_depth = None
                    view_target_depth_mask = None

                    view_depth, view_disparity = self.get_depth(
                        mvd_images[vid].unsqueeze(0),
                        archive_output=False,
                        target_depth=view_target_depth,
                        mask_align=view_target_depth_mask,
                        mask_farther=None,
                        diffusion_steps=30,
                        guidance_steps=6,
                    )
                # heuristic way to scale depth on random view
                # view_depth_mean = view_depth[view_mask.bool()].mean()
                # view_depth[view_mask.bool()] *= valid_depth_mean / view_depth_mean
                # view_depth[view_mask.bool()] = valid_depth_mean

                view_depth, view_mask = align_depth_midas(
                    view_depth,
                    foreground_depth,
                    view_mask.bool(),
                    foreground_mask.bool(),
                )
                view_depth = rearrange(view_depth, "b c h w -> (w h b) c")
                # from pdb import set_trace; set_trace()
                view_points_3d = new_camera_kornia.unproject(self.points, view_depth)
                view_colors = rearrange(
                    mvd_images[vid].unsqueeze(0).to(self.device), "b c h w -> (w h b) c"
                )

                view_mask = rearrange(view_mask, "b c h w -> (w h b) c")[:, 0].bool()
                view_points_3d = view_points_3d[view_mask]
                view_colors = view_colors[view_mask]
                view_normals = view_normals[view_mask]

                # random_mask = torch.rand(view_points_3d.shape[0]).to(self.device)
                # random_mask = (random_mask > 0.2).bool()
                # view_normals = view_normals[random_mask]
                # view_points_3d = view_points_3d[random_mask]
                # view_colors = view_colors[random_mask]

                new_normals = torch.cat([new_normals, view_normals], dim=0)
                new_points_3d = torch.cat([new_points_3d, view_points_3d], dim=0)
                new_colors = torch.cat([new_colors, view_colors.to(self.device)], dim=0)

        if imgto3D and valid_mask is not None:
            img3d_path = self.run_dir / "images" / "img3d"
            if object_id is not None:
                img3d_path = img3d_path / f"object_{object_id}"
            img3d_path.mkdir(exist_ok=True, parents=True)

            input_view_z_mean = new_points_3d[:, 2].mean()
            input_view_z_min = new_points_3d[:, 2].min()

            if 'imgto3d_method' in self.config:
                imgto3d_method = self.config['imgto3d_method']
                if isinstance(imgto3d_method, omegaconf.listconfig.ListConfig):
                    imgto3d_method = imgto3d_method[object_id]
                
                print(f'Using {imgto3d_method} for image to 3d generation')
            else:
                imgto3d_method = 'spar3d'
                print(f'No imgto3d_method specified, using default {imgto3d_method} for image to 3d generation')

            if 'imgto3d_dir' in self.config and os.path.exists(os.path.join(self.config['imgto3d_dir'], f'object_{object_id}.obj')):
                self.config['imgto3d_fg_ratio'] = 0.85
                use_download_mesh = True
            else:
                use_download_mesh = False
            use_download_mesh = False

            if imgto3d_method == 'instantmesh':
                self.config['imgto3d_fg_ratio'] = 0.85
            elif imgto3d_method == 'spar3d':
                self.config['imgto3d_fg_ratio'] = 1 / 1.3
            else:
                raise ValueError(f'Invalid imgto3d_method: {imgto3d_method}')

            object_image, object_mask, update_intrinsics_parameters = (
                process_foreground(image, valid_mask, ratio=self.config['imgto3d_fg_ratio'], res=512)
            )
            # object_depth, _, _ = process_foreground(depth, valid_mask, res=512)

            cv2.imwrite(
                (img3d_path / f"object_image.png").as_posix(),
                (
                    object_image.clone()[0, [2, 1, 0], :, :]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    * 255
                ).astype(np.uint8),
            )

            # object_image_input = object_image[0] * object_mask[0] + 0.5 * (1 - object_mask[0])  # for triposr
            
            # object_image_input = object_image[0] * object_mask[0]  # for instantmesh
            # object_image_input_tensor = object_image_input.unsqueeze(0)
            # object_image_input = Image.fromarray(
            #     (object_image_input.permute(1, 2, 0).cpu().numpy() * 255).astype(
            #         np.uint8
            #     )
            # )

            # needs to have a 4 channel numpy array image result for instantmesh input
            # first 3 channel directly use cv2.imwrite should be BGR
            # last channel is 0-255 mask

            object_image_input_rgb = object_image[0].permute(1, 2, 0).cpu().numpy()
            object_image_input_rgb = (object_image_input_rgb * 255).astype(np.uint8)
            object_image_input_mask = object_mask[0].permute(1, 2, 0).cpu().numpy()
            object_image_input_mask = (object_image_input_mask * 255).astype(np.uint8)
            object_image_input_np = np.concatenate([object_image_input_rgb, object_image_input_mask], axis=-1)
            object_image_input = Image.fromarray(object_image_input_np)
            object_image_input.save((img3d_path / f"object_image_input.png").as_posix())
            
            object_image_input_tensor = object_image_input_rgb * ((object_image_input_mask / 255).astype(np.uint8))
            object_image_input_tensor = torch.from_numpy(object_image_input_tensor).float().to(self.device) / 255
            object_image_input_tensor = object_image_input_tensor.permute(2, 0, 1).unsqueeze(0)


            # scene_codes = self.instantmesh([object_image_input], device=self.device)
            # render_images = self.instantmesh.render(scene_codes, n_views=30, return_type="pil")
            # for ri, render_image in enumerate(render_images[0]):
            #     render_image.save((img3d_path / 'object_render_image_{:03d}.png'.format(ri)).as_posix())

            # # export mesh with vertex color
            # meshes = self.imgto3d.extract_mesh(scene_codes, has_vertex_color=True, resolution=128)
            # mesh = meshes[0]

            # mesh_path = img3d_path / 'mesh'
            # mesh_path.mkdir(exist_ok=True, parents=True)
            # old_verts = mesh.vertices
            # new_verts = old_verts[:, [1, 2, 0]]
            # new_verts[:, 0] *= -1
            # new_verts[:, 2] *= -1
            # mesh.vertices = new_verts
            # mesh.export((mesh_path / 'mesh.obj').as_posix())
            
            if imgto3d_method == 'instantmesh':
                mesh, mv_images = self.get_instantmesh(object_image_input, img3d_path)
                if 'use_dust3r' in self.config:
                    if isinstance(self.config['use_dust3r'], bool) and self.config['use_dust3r']:
                        input_view_rot_x, input_view_rot_y, input_view_rot_z, input_view_c2w = self.dust3r_pose_estimation(img3d_path, object_id)
                    elif isinstance(self.config['use_dust3r'], omegaconf.listconfig.ListConfig) and self.config['use_dust3r'][object_id]:
                        input_view_rot_x, input_view_rot_y, input_view_rot_z, input_view_c2w = self.dust3r_pose_estimation(img3d_path, object_id)
                    else:
                        input_view_rot_x, input_view_rot_y, input_view_rot_z, input_view_c2w = None, None, None, None
                else:
                    input_view_rot_x, input_view_rot_y, input_view_rot_z, input_view_c2w = None, None, None, None

            elif imgto3d_method == 'spar3d':
                mesh, mv_images = self.get_spar3d(object_image_input, img3d_path)
                input_view_rot_x, input_view_rot_y, input_view_rot_z, input_view_c2w = None, None, None, None

            else:
                raise ValueError(f'Invalid imgto3d_method: {imgto3d_method}')

            if use_download_mesh:
                print(f'Loading mesh from {self.config["imgto3d_dir"]}')
                mesh = trimesh.load(os.path.join(self.config['imgto3d_dir'], f'object_{object_id}.obj'))
                mesh.vertices = mesh.vertices[:, [2,0,1]]
                mesh.vertices = mesh.vertices @ np.array([[-1,0,0], [0,1,0], [0,0,-1]])
                mesh.vertices = mesh.vertices @ np.array([[0,0,-1], [-1,0,0], [0,-1,0]])
            
            mv_images = torch.nn.functional.interpolate(
                mv_images,
                size=(
                    object_image_input_tensor.shape[-2],
                    object_image_input_tensor.shape[-1],
                ),
                mode="bilinear",
                align_corners=False,
            )

            verts = mesh.vertices
            faces = mesh.faces
            colors = mesh.visual.vertex_colors
            colors = colors[:, :3] / colors[:, 3:]

            verts_z_min = torch.tensor([verts[:, 2].min()]).to(self.device)
            translate_init = torch.zeros(3).to(self.device)
            translate_init[2] = input_view_z_min * 5.0 * 1000.0
            if self.config['example_name'] == 'kite_2':
                translate_init[2] = input_view_z_min * 1.0 * 1000.0

            optim_3d_save_path = img3d_path / "optimization"
            optim_3d_save_path.mkdir(exist_ok=True, parents=True)

            scale, translation = self.train_with_kpmatching(
                verts,
                faces,
                colors,
                camera,
                image,
                valid_mask,
                image_points_3d,
                optim_3d_save_path,
                translate_init=translate_init,
                image_resolution=512,
                device=self.device,
                input_view_rot_x=input_view_rot_x,
                input_view_rot_y=input_view_rot_y,
                input_view_rot_z=input_view_rot_z,
                input_view_c2w=input_view_c2w,
            )

            # TODO: remove this
            if 'translation_z_scale' in self.config:
                translation_z_scale = self.config['translation_z_scale']
                translation[2] = translation[2] * translation_z_scale

            pts_3d = mesh.vertices
            pts_3d = torch.tensor(pts_3d).float().to(self.device)
            pts_colors = torch.tensor(colors).float().to(self.device)
            pts_normals = torch.rand_like(pts_3d)

            obj_gaussians = self.train_obj_gaussians(
                pts_3d,
                pts_colors,
                pts_normals,
                torch.cat(
                    [object_image_input_tensor, mv_images.to(self.device)], dim=0
                ),
                img3d_path / f"object_{object_id}",
                input_view_rot_x=input_view_rot_x,
                input_view_rot_y=input_view_rot_y,
                input_view_rot_z=input_view_rot_z,
                input_view_c2w=input_view_c2w,
                input_view_mask=object_image_input_mask,
                image_to_3d_method=imgto3d_method,
                input_view_ratio=self.config['imgto3d_fg_ratio']
            )
            # similarly transform gaussians xyz
            if input_view_c2w is not None:
                world_to_camera_input_view = torch.linalg.inv(input_view_c2w)
                gaussians_xyz = obj_gaussians._xyz
                homogenous_coordinates = torch.cat([gaussians_xyz, torch.ones_like(gaussians_xyz[:, :1])], dim=-1)
                transformed_homogenous_coordinates = (world_to_camera_input_view @ homogenous_coordinates.T).T
                transformed_gaussians_xyz = transformed_homogenous_coordinates[:, :3]
                obj_gaussians._xyz = transformed_gaussians_xyz
            obj_gaussians._xyz = (
                obj_gaussians._xyz * scale + translation
            )  # no need to /1000 as in train_data we don't re-scale the gaussians
            obj_gaussians._scaling = obj_gaussians.scaling_inverse_activation(
                obj_gaussians.get_scaling * scale
            )
            self.object_gaussians[object_id] = obj_gaussians
            obj_ys = obj_gaussians._xyz[:, 1]
            if self.objects_y_min is None:
                self.objects_y_min = obj_ys.min()
            else:
                self.objects_y_min = min(self.objects_y_min, obj_ys.min())

            if input_view_c2w is not None:
                homogenous_pts3d = torch.cat([pts_3d, torch.ones_like(pts_3d[:, :1])], dim=-1)
                transformed_homogenous_pts3d = (world_to_camera_input_view @ homogenous_pts3d.T).T
                transformed_pts3d = transformed_homogenous_pts3d[:, :3]
                pts_3d = transformed_pts3d
            
            if self.config['example_name'] == 'food_wine_1' and object_id == 0 and self.config['set_new_object']:
                wine_obj_gaussian = GaussianModel(sh_degree=0)
                wine_obj_gaussian.create_from_pcd(
                    BasicPointCloud(
                        points=self.wine_pts.cpu().numpy(), 
                        colors=self.wine_pts_colors.cpu().numpy(), 
                        normals=self.wine_pts_normals.cpu().numpy()
                    ),
                    self.camera_extent_in_train_data,
                    is_sky=False,
                    is_obj_init=True
                )
                wine_obj_gaussian._xyz = (
                    wine_obj_gaussian._xyz * scale + translation
                )  # no need to /1000 as in train_data we don't re-scale the gaussians
                wine_obj_gaussian._scaling = wine_obj_gaussian.scaling_inverse_activation(
                    wine_obj_gaussian.get_scaling * scale
                )

                # also save the point locations + translations in a txt file for Morph.Points
                # in pytorch3d coordinate
                wine_pts_3d = self.wine_pts.cpu().numpy()
                wine_pts_3d = wine_pts_3d * scale.item() + translation.cpu().numpy()
                wine_pts_3d_dir = img3d_path.as_posix().replace("object_0", "object_2")
                wine_pts_3d_dir = Path(wine_pts_3d_dir)
                wine_pts_3d_dir.mkdir(exist_ok=True, parents=True)
                wine_pts_3d_path = wine_pts_3d_dir / "wine_pts_3d.txt"
                np.savetxt(wine_pts_3d_path.as_posix(), wine_pts_3d)

                # very hacky, as we know there are two objects, so the new liquid is the third object
                self.object_gaussians[1] = None
                self.object_gaussians[2] = wine_obj_gaussian
                self.object_gaussians_faces[1] = None
                self.object_gaussians_faces[2] = None
                self.object_gaussians_meshes_translation[1] = None
                self.object_gaussians_meshes_translation[2] = translation.cpu().numpy()
                self.object_gaussians_meshes[1] = None
                self.object_gaussians_meshes[2] = wine_pts_3d_path.as_posix()
            
            if self.config['example_name'] == 'demo_4' and object_id == 0 and self.config['set_new_object']:
                wine_obj_gaussian = GaussianModel(sh_degree=0)
                wine_obj_gaussian.create_from_pcd(
                    BasicPointCloud(
                        points=self.wine_pts.cpu().numpy(), 
                        colors=self.wine_pts_colors.cpu().numpy(), 
                        normals=self.wine_pts_normals.cpu().numpy()
                    ),
                    self.camera_extent_in_train_data,
                    is_sky=False,
                    is_obj_init=True
                )
                wine_obj_gaussian._xyz = (
                    wine_obj_gaussian._xyz * scale + translation
                )  # no need to /1000 as in train_data we don't re-scale the gaussians
                wine_obj_gaussian._scaling = wine_obj_gaussian.scaling_inverse_activation(
                    wine_obj_gaussian.get_scaling * scale
                )

                # also save the point locations + translations in a txt file for Morph.Points
                # in pytorch3d coordinate
                wine_pts_3d = self.wine_pts.cpu().numpy()
                wine_pts_3d = wine_pts_3d * scale.item() + translation.cpu().numpy()
                wine_pts_3d_dir = img3d_path.as_posix().replace("object_0", "object_2") if self.config['object_num'] == 3 else img3d_path.as_posix().replace("object_0", "object_1")
                wine_pts_3d_dir = Path(wine_pts_3d_dir)
                wine_pts_3d_dir.mkdir(exist_ok=True, parents=True)
                wine_pts_3d_path = wine_pts_3d_dir / "wine_pts_3d.txt"
                np.savetxt(wine_pts_3d_path.as_posix(), wine_pts_3d)

                # very hacky, as we know there are two objects, so the new liquid is the third object
                if self.config['object_num'] == 3:
                    # there will be three objects with the new liquid involved
                    self.object_gaussians[1] = None
                    self.object_gaussians[2] = wine_obj_gaussian
                    self.object_gaussians_faces[1] = None
                    self.object_gaussians_faces[2] = None
                    self.object_gaussians_meshes_translation[1] = None
                    self.object_gaussians_meshes_translation[2] = translation.cpu().numpy()
                    self.object_gaussians_meshes[1] = None
                    self.object_gaussians_meshes[2] = wine_pts_3d_path.as_posix()
                
                elif self.config['object_num'] == 2:
                    # there will be two objects with the new liquid involved
                    self.object_gaussians[1] = wine_obj_gaussian
                    self.object_gaussians_faces[1] = None
                    self.object_gaussians_meshes_translation[1] = translation.cpu().numpy()
                    self.object_gaussians_meshes[1] = wine_pts_3d_path.as_posix()
            
            if self.config['example_name'] == 'to_water_duck' and 'set_new_object' in self.config and self.config['set_new_object']:
                # initialize water points for to_water_duck scene
                # put it here because this involved the background, which requires translation for object
                duck_pts = obj_gaussians._xyz.clone()
                # now in pt3d space, find the x and z range, SAME as the obj_valid_min, obj_valid_max in simulator
                # which is the range of the MPM
                range_x_min = duck_pts[:, 0].min().item()
                range_x_max = duck_pts[:, 0].max().item()
                range_x = range_x_max - range_x_min
                range_z_min = duck_pts[:, 2].min().item()
                range_z_max = duck_pts[:, 2].max().item()
                range_z = range_z_max - range_z_min
                range_x_min_large = range_x_min - range_x
                range_z_min_large = range_z_min - range_z
                range_x_max_large = range_x_max + range_x
                range_z_max_large = range_z_max + range_z

                env_pts_origin = self.get_current_pc_latest()["xyz"] * 1000. # [N, 3]
                env_pts_colors_origin = self.get_current_pc_latest()["rgb"]
                env_pts = env_pts_origin.clone()
                env_pts_colors = env_pts_colors_origin.clone()

                env_pts_x_mask = (env_pts[:, 0] >= range_x_min) & (env_pts[:, 0] <= range_x_max)
                env_pts_z_mask = (env_pts[:, 2] >= range_z_min) & (env_pts[:, 2] <= range_z_max)
                env_pts_x_mask_large = (env_pts[:, 0] >= range_x_min_large) & (env_pts[:, 0] <= range_x_max_large)
                env_pts_z_mask_large = (env_pts[:, 2] >= range_z_min_large) & (env_pts[:, 2] <= range_z_max_large)

                # use large mask to get x and z range
                # use small mask to get y range
                env_pts_mask = env_pts_x_mask & env_pts_z_mask
                env_pts_mask_large = env_pts_x_mask_large & env_pts_z_mask_large

                env_pts_narrow = env_pts[env_pts_mask]
                env_pts = env_pts[env_pts_mask_large]
                env_pts_colors = env_pts_colors[env_pts_mask_large]
                # in x range and z range, use 0.005 as the step size
                # in y range, start from env_pts[:, 1].min() and use step size 0.005 and 5 steps
                # initialize water points, and use closest env_pts colors to initialize water points colors

                # more dense setting
                step_size = 0.01
                y_height = 0.08
                num_y_steps = 40

                # more dynamic setting
                # step_size = 0.005
                # y_height = 0.05
                # num_y_steps = 8 # 10 # 40

                x_coords = torch.arange(range_x_min_large + step_size, range_x_max_large, step_size, device=self.device)
                z_coords = torch.arange(range_z_min_large + step_size, range_z_max_large, step_size, device=self.device)
                y_min = env_pts_narrow[:, 1].min()
                # y_min = env_pts[:, 1].min()
                
                # more dense setting
                y_coords = torch.arange(y_min + y_height / 2, y_min + y_height, y_height / num_y_steps, device=self.device)

                # more dynamic setting
                # y_coords = torch.arange(y_min, y_min + y_height, y_height / num_y_steps, device=self.device)

                # Create meshgrid for x, y, z coordinates
                wX, wY, wZ = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
                
                # Reshape to N x 3 points
                water_points = torch.stack([wX.flatten(), wY.flatten(), wZ.flatten()], dim=1)
                
                # Find closest env point for each water point to get colors
                water_points_expanded = water_points.unsqueeze(1)  # [N, 1, 3]
                env_pts_expanded = env_pts.unsqueeze(0)  # [1, M, 3]
                
                # Calculate distances
                # Split water points into chunks to avoid OOM
                chunk_size = 4096 # 16384
                water_chunks = torch.split(water_points_expanded, chunk_size)
                
                # Initialize distances tensor
                distances = torch.zeros(water_points.shape[0], env_pts.shape[0], device=self.device)
                
                # Process each chunk
                start_idx = 0
                for chunk in water_chunks:
                    # Calculate distances for current chunk
                    chunk_distances = torch.sum((chunk - env_pts_expanded) ** 2, dim=-1)  # [chunk_size, M]
                    
                    # Store chunk distances
                    end_idx = start_idx + chunk.shape[0]
                    distances[start_idx:end_idx] = chunk_distances
                    
                    start_idx = end_idx

                closest_indices = torch.argmin(distances, dim=1)  # [N]
                
                # Get colors from closest env points
                water_colors = env_pts_colors[closest_indices]
                
                # Add simple upward-facing normals
                water_normals = torch.zeros_like(water_points)
                water_normals[:, 1] = 1.0  # Upward facing normals

                self.water_pts = water_points
                self.water_pts_colors = water_colors
                self.water_pts_normals = water_normals
                
                water_obj_gaussian = GaussianModel(sh_degree=0)
                # the scale and translation is already applied in the water_points
                water_obj_gaussian.create_from_pcd(
                    BasicPointCloud(
                        points=self.water_pts.cpu().numpy(),
                        colors=self.water_pts_colors.cpu().numpy(),
                        normals=self.water_pts_normals.cpu().numpy()
                    ),
                    self.camera_extent_in_train_data,
                    is_sky=False,
                    is_obj_init=True
                )

                water_pts_3d = self.water_pts.cpu().numpy()
                water_pts_3d_dir = img3d_path.as_posix().replace("object_0", "object_1")
                water_pts_3d_dir = Path(water_pts_3d_dir)
                water_pts_3d_dir.mkdir(exist_ok=True, parents=True)
                water_pts_3d_path = water_pts_3d_dir / "water_pts_3d.txt"
                np.savetxt(water_pts_3d_path.as_posix(), water_pts_3d)

                self.object_gaussians[1] = water_obj_gaussian
                self.object_gaussians_faces[1] = None
                self.object_gaussians_meshes_translation[1] = translation.cpu().numpy() * 0.
                self.object_gaussians_meshes[1] = water_pts_3d_path.as_posix()
                
            mesh.vertices = pts_3d.cpu().numpy() * scale.item()
            processed_mesh_path = (img3d_path / "mesh_scaled_rotated.obj").as_posix()
            mesh.export(processed_mesh_path)
            self.object_gaussians_meshes[object_id] = processed_mesh_path
            self.object_gaussians_meshes_translation[object_id] = translation.cpu().numpy()

            pts_3d = pts_3d * scale + translation
            pts_3d = pts_3d / 1000.0

            # add new points
            # new_normals = torch.cat([new_normals, pts_normals], dim=0)
            # new_points_3d = torch.cat([new_points_3d, pts_3d], dim=0)
            # new_colors = torch.cat([new_colors, pts_colors], dim=0)

            # replace old points
            new_normals = pts_normals
            new_points_3d = pts_3d
            new_colors = pts_colors

            faces = mesh.faces
            faces = torch.tensor(faces).long().to(self.device)
            self.object_gaussians_faces[object_id] = faces

        self.update_current_pc(
            new_points_3d,
            new_colors,
            normals=new_normals,
            gen_layer=gen_layer,
            object_id=object_id,
            object_mask=valid_mask,
            ground_mask=ground_mask,
        )


        if self.config['example_name'] == 'food_wine_1' and object_id == 1 and 'set_new_object' in self.config and self.config['set_new_object']:
            # update the liquid gaussians as gaussians in the scene at last
            wine_pts_3d = torch.tensor(self.wine_pts).float().to(self.device)
            wine_pts_colors = torch.tensor(self.wine_pts_colors).float().to(self.device)
            wine_pts_normals = self.wine_pts_normals
            self.update_current_pc(
                wine_pts_3d,
                wine_pts_colors,
                normals=wine_pts_normals,
                gen_layer=gen_layer,
                object_id=2,
                object_mask=valid_mask,
                ground_mask=None,
            )
        
        if self.config['example_name'] == 'demo_4' and 'set_new_object' in self.config and self.config['set_new_object']:
            # update the liquid gaussians as gaussians in the scene at last
            if self.config['object_num'] == 3 and object_id == 1:
                wine_pts_3d = torch.tensor(self.wine_pts).float().to(self.device)
                wine_pts_colors = torch.tensor(self.wine_pts_colors).float().to(self.device)
                wine_pts_normals = self.wine_pts_normals
                self.update_current_pc(
                    wine_pts_3d,
                    wine_pts_colors,
                    normals=wine_pts_normals,
                    gen_layer=gen_layer,
                    object_id=2,
                    object_mask=valid_mask,
                    ground_mask=None,
                )
            elif self.config['object_num'] == 2 and object_id == 0:
                wine_pts_3d = torch.tensor(self.wine_pts).float().to(self.device)
                wine_pts_colors = torch.tensor(self.wine_pts_colors).float().to(self.device)
                wine_pts_normals = self.wine_pts_normals
                self.update_current_pc(
                    wine_pts_3d,
                    wine_pts_colors,
                    normals=wine_pts_normals,
                    gen_layer=gen_layer,
                    object_id=1,
                    object_mask=valid_mask,
                    ground_mask=None,
                )
        
        if self.config['example_name'] == 'to_water_duck' and object_id == 0 and 'set_new_object' in self.config and self.config['set_new_object']:
            # update the water gaussians as gaussians in the scene at last
            water_pts_3d = torch.tensor(self.water_pts).float().to(self.device)
            water_pts_colors = torch.tensor(self.water_pts_colors).float().to(self.device)
            water_pts_normals = self.water_pts_normals
            self.update_current_pc(
                water_pts_3d,
                water_pts_colors,
                normals=water_pts_normals,
                gen_layer=gen_layer,
                object_id=1,
                object_mask=valid_mask,
                ground_mask=None,
            )
        return new_points_3d, new_colors

    @torch.enable_grad()
    def train_obj_gaussians(
        self,
        pts_3d,
        pts_colors,
        pts_normals,
        images,
        optim_dir,
        input_view_rot_x=None,
        input_view_rot_y=None,
        input_view_rot_z=None,
        input_view_c2w=None,
        input_view_mask=None,
        image_to_3d_method="spar3d",
        input_view_ratio=1.0
    ):
        optim_dir.mkdir(exist_ok=True, parents=True)
        pts_3d = pts_3d.permute(1, 0).cpu().numpy()
        pts_colors = pts_colors.cpu().numpy()
        pts_normals = pts_normals.cpu().numpy()
        W, H = 512, 512
        camera_angle_x = 2 * np.arctan(W / (2 * self.init_focal_length))
        opt = GSParams()

        if image_to_3d_method == 'instantmesh':
            azimuths = self.instantmesh_azimuth
            elevations = self.instantmesh_elevation
            radius_set = self.instantmesh_radius
        elif image_to_3d_method == 'spar3d':
            azimuths = self.spar3d_render_azimuth
            elevations = self.spar3d_render_elevation
            radius_set = self.spar3d_render_radius
        else:
            raise ValueError(f'Invalid image_to_3d_method: {image_to_3d_method}')

        # Add input view rotation angles to the beginning of azimuths and elevations arrays
        if input_view_rot_y is not None:
            azimuths = np.concatenate([[input_view_rot_y.item()], azimuths])
            elevations = np.concatenate([[input_view_rot_x.item()], elevations])
        else:
            azimuths = np.concatenate([[0], azimuths])
            elevations = np.concatenate([[0], elevations])
        # now the train_with_kpmatching only gets the scale and translation, here the mesh is still in canonical space
        # for optimization, we first use the dust3r output camera with other canonical view cameras
        # after optimization, we further rotate the mesh to match the input image

        frames = []
        for i, image in enumerate(images):
            radius = radius_set
            image = image.unsqueeze(0)

            # input view crop the image to be center-aligned
            if i == 0:
                input_view_mask = torch.tensor(input_view_mask).to(self.device).permute(2, 0, 1).unsqueeze(0)
                input_view_mask = (input_view_mask / 255).int()
                h_valid, w_valid = torch.where(input_view_mask == 1)[2:4]
                h_valid_min, h_valid_max = h_valid.min(), h_valid.max()
                w_valid_min, w_valid_max = w_valid.min(), w_valid.max()
                # Get dimensions of valid region
                h_valid_size = h_valid_max - h_valid_min
                w_valid_size = w_valid_max - w_valid_min
                
                # Use larger dimension for square size
                square_size = max(h_valid_size, w_valid_size)
                
                # Calculate padding needed for smaller dimension
                h_pad = max(0, square_size - h_valid_size) 
                w_pad = max(0, square_size - w_valid_size)
                
                # Pad half on each side
                h_pad_top = h_pad // 2
                h_pad_bottom = h_pad - h_pad_top
                w_pad_left = w_pad // 2 
                w_pad_right = w_pad - w_pad_left
                
                # Crop and pad to square
                image_cropped = image[:, :, h_valid_min:h_valid_max, w_valid_min:w_valid_max]
                image = torch.nn.functional.pad(
                    image_cropped,
                    (w_pad_left, w_pad_right, h_pad_top, h_pad_bottom),
                    mode='constant',
                    value=0
                )
                # resize to 512x512
                image = torch.nn.functional.interpolate(
                    image,
                    size=(512, 512),
                    mode='bilinear',
                    align_corners=False,
                )

            image_pil = ToPILImage()(image[0])

            # transform_matrix_pt3d = self.cameras_archive[i].get_world_to_view_transform().get_matrix()[0]
            # transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)
            # transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()
            opengl_to_pt3d = torch.diag(
                torch.tensor([-1.0, 1, -1, 1], device=self.device)
            )
            # transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            if i != 0:
                xx = (
                    -1
                    * radius
                    * np.cos(np.deg2rad(elevations[i]))
                    * np.sin(np.deg2rad(azimuths[i]))
                )
                yy = radius * np.sin(np.deg2rad(elevations[i]))
                zz = (
                    -1
                    * radius
                    * np.cos(np.deg2rad(elevations[i]))
                    * np.cos(np.deg2rad(azimuths[i]))
                )
                cam_locations = np.stack([xx, yy, zz], axis=-1)
                cam_locations = torch.from_numpy(cam_locations).float().to(self.device)

                # z-forward, y-upward, x-left
                z_axis = torch.zeros_like(cam_locations) - cam_locations
                z_axis = F.normalize(z_axis, dim=-1).float()
                x_axis = torch.linalg.cross(
                    torch.tensor([0, 1, 0], dtype=torch.float32, device=self.device),
                    z_axis,
                    dim=-1,
                )
                x_axis = F.normalize(x_axis, dim=-1).float()
                y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
                y_axis = F.normalize(y_axis, dim=-1).float()

                extrinsics = torch.stack([x_axis, y_axis, z_axis, cam_locations], dim=-1)
                extrinsics = torch.cat(
                    [extrinsics, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).to(self.device)],
                    dim=0,
                )
                transform_matrix_c2w_pt3d = extrinsics
            else:
                if input_view_c2w is not None:
                    transform_matrix_c2w_pt3d = input_view_c2w
                else:
                    transform_matrix_c2w_pt3d = torch.tensor([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, -self.instantmesh_radius if image_to_3d_method == 'instantmesh' else -self.spar3d_render_radius],
                        [0, 0, 0, 1]
                    ]).to(self.device)
            transform_matrix_c2w_opengl = transform_matrix_c2w_pt3d @ opengl_to_pt3d

            transform_matrix = transform_matrix_c2w_opengl.cpu().numpy().tolist()
            frame = {
                "image": image_pil,
                "transform_matrix": transform_matrix,
                "no_loss_mask": None,
            }
            frames.append(frame)

        train_data = {
            "frames": frames,
            "pcd_points": pts_3d,
            "pcd_colors": pts_colors,
            "pcd_normals": pts_normals,
            "camera_angle_x": camera_angle_x,
            "W": W,
            "H": H,
        }

        obj_gaussians = GaussianModel(sh_degree=0)
        obj_scene = Scene(train_data, obj_gaussians, opt, is_obj_init=True)
        self.camera_extent_in_train_data = obj_scene.cameras_extent

        trainCameras = obj_scene.getTrainCameras().copy()
        obj_gaussians.compute_3D_filter(cameras=trainCameras, initialize_scaling=True)

        for iter in range(self.config['imgto3d_optimization_iters']):
            viewpoint_stack = obj_scene.getTrainCameras().copy()
            rand_idx = np.random.randint(0, len(viewpoint_stack) - 1)

            # for spar3d, mostly optimize towards the first camera is enough
            # as it just reconstructs the foreground, implemented this for demo_3 and after
            if image_to_3d_method == 'spar3d':
                if np.random.rand() < 0.9:
                    rand_idx = 0
                else:
                    rand_idx = np.random.randint(0, len(viewpoint_stack) - 2)
                    rand_idx += 1
                
                for param_group in obj_gaussians.optimizer.param_groups:
                    if param_group["name"] == "f_dc":
                        param_group["lr"] = 0.01    # for spar3d, it's absolutely aligned so optimize the color

            if self.config['example_name'] == 'demo_9':
                for param_group in obj_gaussians.optimizer.param_groups:
                    if param_group["name"] == "opacity":
                        param_group["lr"] = 0.0
            
            if rand_idx == 0:
                train_background = torch.tensor(
                    [0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"
                )
            else:
                if image_to_3d_method == 'instantmesh':
                    train_background = torch.tensor(
                        [1.0, 1.0, 1.0], dtype=torch.float32, device="cuda"
                    )
                else:
                    train_background = torch.tensor(
                        [0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"
                    )
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            render_pkg = render(viewpoint_cam, obj_gaussians, opt, train_background)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            if rand_idx == 0:
                mask = render_pkg['final_opacity'].unsqueeze(0)
                mask = mask != 0
                mask = mask.int()
                # also crop out the image from this mask
                h_valid, w_valid = torch.where(mask == 1)[2:4]
                h_valid_min, h_valid_max = h_valid.min(), h_valid.max()
                w_valid_min, w_valid_max = w_valid.min(), w_valid.max()
                # Get dimensions of valid region
                h_valid_size = h_valid_max - h_valid_min
                w_valid_size = w_valid_max - w_valid_min
                
                # Use larger dimension for square size
                square_size = max(h_valid_size, w_valid_size)
                
                # Calculate padding needed for smaller dimension
                h_pad = max(0, square_size - h_valid_size) 
                w_pad = max(0, square_size - w_valid_size)
                
                # Pad half on each side
                h_pad_top = h_pad // 2
                h_pad_bottom = h_pad - h_pad_top
                w_pad_left = w_pad // 2 
                w_pad_right = w_pad - w_pad_left
                
                # Crop and pad to square
                image = image.unsqueeze(0)
                image_cropped = image[:, :, h_valid_min:(h_valid_max+1), w_valid_min:(w_valid_max+1)]
                image = torch.nn.functional.pad(
                    image_cropped,
                    (w_pad_left, w_pad_right, h_pad_top, h_pad_bottom),
                    mode='constant',
                    value=0
                )
                # resize to 512x512
                image = torch.nn.functional.interpolate(
                    image,
                    size=(512, 512),
                    mode='bilinear',
                    align_corners=False,
                )
                image = image.squeeze(0)

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0 - ssim(image, gt_image)
            )
            loss.backward()
            obj_gaussians.optimizer.step()
            obj_gaussians.optimizer.zero_grad(set_to_none=True)

            if iter % 100 == 0:
                image_np = image.clone().detach().cpu().permute(1, 2, 0).numpy()
                image_np = (image_np[:, :, [2, 1, 0]] * 255).astype(np.uint8)
                gt_np = gt_image.clone().detach().cpu().permute(1, 2, 0).numpy()
                gt_np = (gt_np[:, :, [2, 1, 0]] * 255).astype(np.uint8)
                cv2.imwrite(optim_dir / f"iter_{iter:03d}_render.png", image_np)
                cv2.imwrite(optim_dir / f"iter_{iter:03d}_gt.png", gt_np)

        return obj_gaussians

    @torch.enable_grad()
    def train_with_kpmatching(
        self,
        verts,
        faces,
        colors,
        cameras,
        gt_rgb,
        gt_mask,
        gt_points,
        save_path,
        translate_init=None,
        image_resolution=512,
        device="cuda",
        input_view_rot_x=None,
        input_view_rot_y=None,
        input_view_rot_z=None,
        input_view_c2w=None,
    ):
        # quant_num = 4
        # quant = [x / quant_num for x in range(1,quant_num)]
        if 'kp_quant' in self.config:
            quant = self.config['kp_quant']
        else:
            quant = [0.05, 0.95]
        quant = torch.tensor(quant).float().to(device)

        if 'kp_per_quant' in self.config:
            per_quant = self.config['kp_per_quant']
        else:
            per_quant = [0.15, 0.85]
        per_quant = torch.tensor(per_quant).float().to(device)

        gt_kp_h, gt_kp_w = kps_from_quants(gt_mask, quant, per_quant=per_quant)

        # visualize on gt image
        gt_save_path = (save_path / "gt_kps.png").as_posix()
        save_mask_kps(gt_mask, gt_kp_h, gt_kp_w, gt_save_path)

        verts = torch.tensor(verts).float().unsqueeze(0).to(device)
        if input_view_c2w is not None:
            world_to_camera_input_view = torch.linalg.inv(input_view_c2w)
            homogenous_vertices = torch.cat([verts[0], torch.ones_like(verts[0, :, :1])], dim=-1)
            transformed_homogenous_vertices = (world_to_camera_input_view @ homogenous_vertices.T).T
            transformed_vertices = transformed_homogenous_vertices[:, :3]
            verts = transformed_vertices.unsqueeze(0)

        # verts = verts + torch.tensor([0., 0., 5.]).unsqueeze(0).to(device)
        faces = torch.tensor(faces).long().to(device).unsqueeze(0)
        # colors = torch.tensor(colors).float().to(device).unsqueeze(0)
        # colors = colors[..., [2,1,0]]
        verts_min = verts.min(dim=1)[0].unsqueeze(1)
        verts_max = verts.max(dim=1)[0].unsqueeze(1)
        colors = (verts.clone() - verts_min) / (
            verts_max - verts_min
        )  # color is bounded with original mesh coordinates

        verts = verts + translate_init  # move the mesh so it's visible in the camera

        textures = Textures(verts_rgb=colors)
        mesh = Meshes(verts, faces, textures)

        lights = PointLights(
            ambient_color=((0.0, 0.0, 0.0),),
            diffuse_color=((0.0, 0.0, 0.0),),
            specular_color=((0.0, 0.0, 0.0),),
            device=device,
            location=[[0.0, 0.0, -3.0]],
        )
        raster_settings = RasterizationSettings(
            image_size=image_resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        # TODO: debug this
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardShader(device=device, cameras=cameras),
        )

        image_predict = renderer(mesh)
        rgb = image_predict[..., :3].permute(0, 3, 1, 2)
        mask = image_predict[..., 3:].permute(0, 3, 1, 2)
        rgb = rgb * mask

        mesh_kp_h, mesh_kp_w = kps_from_quants(mask, quant, per_quant=per_quant)
        mesh_save_path = (save_path / "mesh_kps.png").as_posix()
        save_mask_kps(mask, mesh_kp_h, mesh_kp_w, mesh_save_path)
        # save_mask_kps(rgb, mesh_kp_h, mesh_kp_w, mesh_save_path)

        gt_kps = gt_points[:, :, gt_kp_h, gt_kp_w][0].permute(1, 0)
        mesh_kps = rgb[:, :, mesh_kp_h, mesh_kp_w][0].permute(1, 0)
        mesh_kps = mesh_kps * (verts_max[0] - verts_min[0]) + verts_min[0]

        gt_kps = 1000.0 * gt_kps
        # now, the scale and translation values can be solved using least square
        # they have relation: s * mesh_kps + [t1,t2,t3] = gt_kps
        A = mesh_kps
        B = gt_kps.flatten().unsqueeze(-1)
        # [K, 3, 4]
        A_compact = torch.cat(
            [
                A.unsqueeze(-1),
                torch.eye(3)
                .unsqueeze(0)
                .repeat(mesh_kps.shape[0], 1, 1)
                .to(device=device),
            ],
            dim=-1,
        )
        A_compact_final = torch.cat([i for i in A_compact], dim=0)

        solution = torch.linalg.lstsq(A_compact_final, B).solution
        scale = solution[0]
        translation = solution[1:, 0]

        return scale, translation

    # @torch.no_grad()
    # def remove_floating_points(self):
    #     self.current_pc = None
    #     assert len(self.points_3d_list) == len(self.colors_list) == len(self.floating_point_mask_list)
    #     for points_3d, colors, floating_point_mask in zip(self.points_3d_list, self.colors_list, self.floating_point_mask_list):
    #         points_3d = points_3d[floating_point_mask]
    #         colors = colors[floating_point_mask]
    #         self.update_current_pc(points_3d, colors)


class KeyframeGen(FrameSyn):
    def __init__(
        self,
        config,
        inpainter_pipeline,
        depth_model,
        mask_generator,
        segment_model=None,
        segment_processor=None,
        normal_estimator=None,
        rotation_path=None,
        inpainting_resolution=None,
        mvdiffusion=None,
        sam_model=None,
    ):
        """This class is for generating keyframes. It inherits from FrameSyn. It implements the following tasks:
        1. Render
        2. Set cameras
        3. Initialize point cloud
        4. Post-process depth
        """
        super().__init__(
            config,
            inpainter_pipeline=inpainter_pipeline,
            depth_model=depth_model,
            normal_estimator=normal_estimator,
            mvdiffusion=mvdiffusion,
            sam_model=sam_model,
        )

        ####### Set up placeholder attributes #######

        ####### Set up archives #######
        self.rendered_image_latest = torch.zeros(1, 3, 512, 512)
        self.rendered_depth_latest = torch.zeros(1, 1, 512, 512)
        self.no_loss_mask_latest = torch.zeros(1, 1, 512, 512).bool()
        self.no_loss_mask_latest_layer = torch.zeros(1, 1, 512, 512).bool()
        self.current_camera = None

        self.rendered_images = []
        self.rendered_depths = []
        self.no_loss_masks = (
            []
        )  # Indicating which pixels to remove for foreground in optimizing 3DGS
        self.no_loss_masks_layer = (
            []
        )  # Indicating which pixels to remove for layer in optimizing 3DGS

        ####### Set up attributes #######
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        run_dir_root = Path(config["runs_dir"])
        self.run_dir = run_dir_root / f"Gen-{dt_string}"
        self.logger = SimpleLogger(self.run_dir / "log.txt")
        self.mask_generator = mask_generator
        self.segment_model = segment_model
        self.segment_processor = segment_processor
        self.sky_hard_depth = config["sky_hard_depth"]
        self.sky_erode_kernel_size = config["sky_erode_kernel_size"]
        self.is_upper_mask_aggressive = False

        self.rotation_range_theta = config["rotation_range"]
        self.interp_frames = config["frames"]
        self.camera_speed = config["camera_speed"]
        self.camera_speed_multiplier_rotation = config[
            "camera_speed_multiplier_rotation"
        ]

        ####### Initialization functions #######
        (self.run_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "images" / "knn_masks").mkdir(exist_ok=True, parents=True)
        # (self.run_dir / 'images' / "floating_point_mask").mkdir(exist_ok=True, parents=True)
        (self.run_dir / "images" / "depth_should_be").mkdir(exist_ok=True, parents=True)
        (self.run_dir / "images" / "depth_conditioned").mkdir(
            exist_ok=True, parents=True
        )
        # (self.run_dir / 'images' / "floating_masked_images").mkdir(exist_ok=True, parents=True)
        (self.run_dir / "images" / "layer").mkdir(exist_ok=True, parents=True)
        (self.run_dir / "images" / "disparity_gradient").mkdir(
            exist_ok=True, parents=True
        )
        (self.run_dir / "images" / "multiview_diffusion").mkdir(
            exist_ok=True, parents=True
        )

        # rotation matrix of each scene
        self.scene_cameras_idx = []
        self.center_camera_idx = None
        self.generate_cameras(rotation_path)
        self.cameras_users = []
        self.inpainting_resolution = inpainting_resolution

    @torch.no_grad()
    def get_camera_at_origin(self):
        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = self.init_focal_length
        K[0, 1, 1] = self.init_focal_length
        K[0, 0, 2] = 256
        K[0, 1, 2] = 256
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        camera = PerspectiveCameras(
            K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device
        )
        return camera

    @torch.no_grad()
    def recompose_image_latest_and_set_current_pc(self):
        self.set_current_camera(self.get_camera_at_origin(), archive_camera=True)
        sem_map = self.update_sky_mask()
        render_output = self.render(render_sky=True)

        '''
        Now the self.image_latest is using rendered sky from gaussians to replace original sky,
        should be pretty much the same.
        '''
        self.image_latest = soft_stitching(
            render_output["rendered_image"], self.image_latest, self.sky_mask_latest
        )  # Replace generated sky with rendered sky

        (self.run_dir / "segmentation").mkdir(exist_ok=True, parents=True)

        '''
        Here we'll further run RepViT on the image_latest, and save the segmentation result,
        this is actually not useful in following objects decomposition, as we tend to use SAM now.
        '''
        save_sem_map(sem_map, self.image_latest, self.run_dir / "segmentation")

        ground_mask = self.generate_ground_mask(sem_map=sem_map)[None, None]
        # ground_mask = self.generate_ground_mask_SAM(ground_ids=self.config["ground_ids"])[None, None]
        depth_should_be_ground = self.compute_ground_depth(camera_height=0.0003)
        ground_outputable_mask = (depth_should_be_ground > 0.001) & (
            depth_should_be_ground < 0.006 * 0.8
        )

        with torch.no_grad():
            depth_guided, _ = self.get_depth(
                self.image_latest,
                archive_output=True,
                target_depth=depth_should_be_ground,
                mask_align=(ground_mask & ground_outputable_mask),
                diffusion_steps=30,
                guidance_steps=8,
            )
        self.refine_disp_with_segments(
            no_refine_mask=ground_mask.squeeze().cpu().numpy()
        )
        # # Visualize depth_should_be
        save_depth_map(self.depth_latest[0, 0].cpu().numpy(), self.run_dir / f"{self.kf_idx:02d}_depth.png", vmax=0.006, vmin=0)
        # save_depth_map(depth_should_be_ground[0, 0].cpu().numpy(), self.run_dir / f"{self.kf_idx:02d}_ground_depth_should_be.png", vmax=0.006, vmin=0)
        # save_depth_map(depth_guided[0, 0].cpu().numpy(), self.run_dir / f"{self.kf_idx:02d}_ground_depth_guided.png", vmax=0.006, vmin=0)
        # with torch.no_grad():
        #     depth_original, _ = self.get_depth(self.image_latest, archive_output=False)
        # save_depth_map(depth_original[0, 0].cpu().numpy(), self.run_dir / f"{self.kf_idx:02d}_depth_original.png", vmax=0.006, vmin=0)
        # # Visualize mask_align
        # ToPILImage()(ground_mask[0,0].float()).save(self.run_dir / f"{self.kf_idx:02d}_ground_mask_align.png")
        # guided_loss = l1_loss(depth_guided, depth_should_be_ground, no_loss_mask=~ground_mask)
        # original_loss = l1_loss(depth_original, depth_should_be_ground, no_loss_mask=~ground_mask)
        # self.logger.print(f"[00 Ground] Guided loss: {guided_loss.item()}, Original loss: {original_loss.item()}")

        if self.config["gen_layer"]:
            if (
                "foreground_ids" not in self.config.keys()
                or len(self.config["foreground_ids"]) == 0
            ):
                print(
                    f"check {self.run_dir} for segmentation results and specify foreground_ids in config!"
                )
                raise NotImplementedError()
            if "ground_ids" not in self.config.keys():
                ground_ids = ["3", "6", "9", "11", "13", "26", "29", "46", "52", "128"]
            else:
                ground_ids = self.config["ground_ids"]
            
            '''
            In here, we use SAM masks to generate self.mask_disocclusion,
            and self.depth_latest and self.image_latest inpainted or pre-inpainted.
            '''

            self.generate_layer(
                pred_semantic_map=sem_map,
                foreground_ids=self.config["foreground_ids"],
                ground_ids=ground_ids,
            )
            depth_should_be = self.depth_latest_init
            mask_to_align_depth = ~(self.mask_disocclusion.bool()) & (
                depth_should_be < 0.006 * 0.8
            )
            mask_to_farther_depth = self.mask_disocclusion.bool() & (
                depth_should_be < 0.006
            )
            with torch.no_grad():
                self.depth, self.disparity = self.get_depth(
                    self.image_latest,
                    archive_output=True,
                    target_depth=depth_should_be,
                    mask_align=mask_to_align_depth,
                    mask_farther=mask_to_farther_depth,
                    diffusion_steps=20,
                    guidance_steps=6,
                )
            self.refine_disp_with_segments(
                no_refine_mask=ground_mask.squeeze().cpu().numpy(),
                keep_threshold_disp_range=-1,  # actually don't refine
                existing_mask=~(self.mask_disocclusion).bool().squeeze().cpu().numpy(),
                existing_disp=self.disparity_latest_init.squeeze().cpu().numpy(),
            )

            if "without_heuristic_set_depth_image" in self.config.keys() and self.config["without_heuristic_set_depth_image"]:
                print("no heuristic reset depth and image for in-painting")
            else:
                depth_reset = heruistic_reset_depth(
                    self.depth_latest,
                    dilation(self.mask_disocclusion, kernel=torch.ones(10, 10).cuda()),
                    method="mean",
                )
                self.depth_latest = depth_reset
                image_reset = heruistic_reset_depth(
                    self.image_latest,
                    dilation(self.mask_disocclusion, kernel=torch.ones(20, 20).cuda()),
                    method="mean",
                )
                self.image_latest = image_reset
            
            wrong_depth_mask = self.depth_latest < self.depth_latest_init
            self.depth_latest[wrong_depth_mask] = (
                self.depth_latest_init[wrong_depth_mask] + 0.0001
            )

            save_depth_map(self.depth_latest[0, 0].cpu().numpy(), self.run_dir / f"{self.kf_idx:02d}_depth_inpainted.png", vmax=0.006, vmin=0)

            # self.depth_latest = self.mask_disocclusion * self.depth_latest + (1-self.mask_disocclusion) * self.depth_latest_init
            self.update_sky_mask()
            self.update_sky_mask()
            self.update_current_pc_by_kf(
                image=self.image_latest,
                depth=self.depth_latest,
                valid_mask=~self.sky_mask_latest,
                ground_mask=(ground_mask & ground_outputable_mask),
            )  # Base only

            if 'align_y_translation' in self.config and self.config['align_y_translation']:
                # TODO: automate this
                # [1, 1, 512, 512]

                if self.config['example_name'] == "domino":
                    depth_before = self.depth_latest_init.clone()
                    obj_mask = self.object_masks[0].bool() | self.object_masks[1].bool() | self.object_masks[2].bool() | self.object_masks[3].bool()
                    depth_anchor = depth_before[obj_mask].mean()
                    depth_before[obj_mask] = depth_anchor
                    self.depth_latest_init = depth_before

                if self.config['example_name'] == "smoke_interaction_2":
                    # simply make the depth of the smoke closer, with the same translation to all the pixels
                    depth_before = self.depth_latest_init.clone()
                    depth_anchor = depth_before[self.object_masks[0].bool()]
                    depth_anchor = depth_anchor - 0.001
                    depth_before[self.object_masks[0].bool()] = depth_anchor
                    self.depth_latest_init = depth_before
                
                elif self.config['example_name'] == "demo_6":
                    depth_before = self.depth_latest_init.clone()
                    depth_before[self.object_masks[0].bool()] = 0.0011
                    self.depth_latest_init = depth_before
                
                elif self.config['example_name'] == "demo_7":
                    depth_before = self.depth_latest_init.clone()
                    hair_mask = torch.logical_or(self.object_masks[1].bool(), self.object_masks[2].bool())
                    depth_before_hair = depth_before[hair_mask].mean() * 0.72
                    depth_before[hair_mask] = depth_before_hair

                    depth_before_hat = depth_before[self.object_masks[0].bool()].mean()
                    # with previous inpainting image
                    # depth_before[self.object_masks[0].bool()] += (depth_before_hair - depth_before_hat)

                    # with no hair inpainting image
                    depth_before[self.object_masks[0].bool()] += (depth_before_hair - depth_before_hat) * 0.3
                    self.depth_latest_init = depth_before

                else:
                    anchor_id = 0
                    depth_before = self.depth_latest_init.clone()
                    depth_anchor_mean = self.depth_latest_init[self.object_masks[anchor_id].bool()].mean()

                    for object_id in range(len(self.object_masks)):
                        if object_id == anchor_id:
                            continue
                        mask_update = self.object_masks[object_id].bool()

                        depth_mean_now = depth_before[mask_update].mean()
                        depth_diff = depth_anchor_mean - depth_mean_now
                        depth_before[mask_update] += depth_diff * 1.1
                
                    self.depth_latest_init = depth_before

            object_num = self.config["object_num"]
            (self.run_dir / "segmentation").mkdir(exist_ok=True, parents=True)
            for object_id in range(len(self.object_masks)):
                object_mask_binary = (self.object_masks[object_id][0, 0].cpu().numpy().astype(np.uint8) * 255)
                cv2.imwrite((self.run_dir / f"segmentation/object_{object_id:02d}.png").as_posix(), object_mask_binary)
                object_mask = self.object_masks[object_id]
                self.update_current_pc_by_kf(
                    image=self.image_latest_init,
                    depth=self.depth_latest_init,
                    valid_mask=object_mask,
                    gen_layer=True,
                    multiview=False,
                    imgto3D=self.config["imgto3D"],
                    object_id=object_id,
                )  # Object layer
            self.object_masks = []
        else:
            self.update_current_pc_by_kf(
                image=self.image_latest,
                depth=self.depth_latest,
                valid_mask=~self.sky_mask_latest,
            )
        self.archive_latest()

    @torch.no_grad()
    def compute_ground_depth(self, camera_height=0.0003):
        """
        Compute the depth map in PyTorch, assuming that after camera unproject, all pixels will lie in the XoZ plane.
        return:
            analytic_depth: [1, 1, 512, 512] torch tensor containing depth values
        """
        focal_length = self.init_focal_length
        x_res, y_res = 512, 512
        y_principal = 256

        # Generate a grid of y-coordinate values aligned directly with its use in the final tensor
        y_grid = torch.arange(y_res).view(1, 1, y_res, 1)

        # Compute the depth using the formula d = h * f / (y - p_y)
        denominator = torch.where(
            y_grid - y_principal != 0, y_grid - y_principal, torch.tensor(1e-10)
        )
        depth_map = (camera_height * focal_length) / denominator

        # Explicitly expand the last dimension to match x_res
        depth_map = depth_map.expand(-1, -1, -1, x_res)

        return depth_map.to(self.device)

    def generate_sky_pointcloud(
        self,
        syncdiffusion_model: SyncDiffusion = None,
        image=None,
        mask=None,
        gen_sky=False,
        style=None,
    ):
        image_height = 512
        image_width = 6144
        w_start = 256
        stride = 16
        anchor_view_idx = w_start // 8 // stride
        layers_panorama = 2
        num_inference_steps = 50
        guidance_scale = 7.5
        sync_weight = 80.0
        sync_decay_rate = 0.98
        sync_freq = 3
        sync_thres = 50

        example_name = self.config["example_name"]

        def linear_blend(images, overlap=100):

            # create blending field
            alpha = np.linspace(0, 1, overlap).reshape(overlap, 1, 1)

            for i, img in enumerate(images):
                img_new = np.array(img)
                if i != 0:
                    overlap_img2 = img_new[512 - overlap :, :, :]
                    top_img = img_new[: 512 - overlap, :, :]
                    blend_overlap = overlap_img1 * (1 - alpha) + overlap_img2 * alpha

                    # combine the image
                    blended_image = np.concatenate(
                        (top_img, blend_overlap, bottom_img), axis=0
                    )
                    img_old = blended_image
                else:
                    img_old = img_new

                overlap_img1 = img_old[:overlap, :, :]
                bottom_img = img_old[overlap:, :, :]

            blended_image = (blended_image).astype(np.uint8)
            return Image.fromarray(blended_image)

        imgs = []
        gen_layer_0 = (
            not os.path.exists(f"./sky_image/{example_name}/sky_0.png")
        ) or gen_sky
        gen_layer_1 = (
            (not os.path.exists(f"./sky_image/{example_name}/sky_1.png"))
            or gen_layer_0
            or gen_sky
        )
        gen_layer_2 = (
            (not os.path.exists(f"./sky_image/{example_name}/sky_2.png"))
            or gen_layer_1
            or gen_sky
        )

        for layer in range(layers_panorama):
            if layer == 0:
                if gen_layer_0:
                    init_image = torch.zeros((1, 3, image_height, image_width))
                    init_image[:, :, :, w_start : w_start + image_height] = image
                    init_image = init_image.to(self.device)
                    ToPILImage()(init_image[0]).save(
                        self.run_dir / f"{layer:02d}_init_image.png"
                    )

                    mask_image = torch.ones((1, 1, image_height, image_width))
                    mask_image[:, :, :, w_start : w_start + image_height] = 1 - mask
                    mask_image = mask_image.to(self.device)
                    ToPILImage()(mask_image.float()[0]).save(
                        self.run_dir / f"{layer:02d}_mask.png"
                    )

                    # Inpaint init_image using inpaint_cv2()
                    mask_image_eroded = dilation(
                        mask_image, kernel=torch.ones(10, 10).cuda()
                    )
                    init_image = inpaint_cv2(init_image, mask_image_eroded)
                    init_image = init_image.to(self.device)
                    ToPILImage()(init_image[0]).save(
                        self.run_dir / f"{layer:02d}_inpainted_init_image.png"
                    )

                    # Block-expand mask using an aggresive way
                    mask_ = (mask_image[0, 0].cpu().numpy() * 255).astype(np.uint8)
                    mask_block_size = 8
                    mask_ = skimage.measure.block_reduce(
                        mask_, (mask_block_size, mask_block_size), np.min
                    )
                    mask_ = mask_.repeat(mask_block_size, axis=0).repeat(
                        mask_block_size, axis=1
                    )
                    mask_image = ToTensor()(mask_).unsqueeze(0).to(self.device)
                    ToPILImage()(mask_image.float()[0]).save(
                        self.run_dir / f"{layer:02d}_mask_blocky.png"
                    )
                else:
                    img = Image.open(f"./sky_image/{example_name}/sky_0.png")
                    # img.save(self.run_dir / f"{layer:02d}_synced_output.png")
                    imgs.append(img)
                    continue
            else:
                if gen_layer_1:
                    init_image = imgs[-1]
                    init_image = ToTensor()(init_image).unsqueeze(0).to(self.device)
                    toprows = init_image[:, :, :100, :]
                    remaining = init_image[:, :, 100:, :]
                    init_image = torch.cat((remaining, toprows), dim=-2)
                    ToPILImage()(init_image[0]).save(
                        self.run_dir / f"{layer:02d}_init_image.png"
                    )

                    mask_image = torch.ones((1, 1, image_height, image_width))
                    mask_image[:, :, -100:, :] = 0
                    mask_image = mask_image.to(self.device)
                    ToPILImage()(mask_image.float()[0]).save(
                        self.run_dir / f"{layer:02d}_mask.png"
                    )
                else:
                    img = Image.open(f"./sky_image/{example_name}/sky_1.png")
                    # img.save(self.run_dir / f"{layer:02d}_synced_output.png")
                    imgs.append(img)
                    continue

            print(f"[INFO] generating sky layer {layer} ...")
            prompts = (
                f"sky, blue sky, horizon, distant hills. style: {style}"
                if layer == 0
                else f"sky, blue sky, cloud. style: {style}"
            )
            img = syncdiffusion_model.sample(
                prompts=prompts,
                negative_prompts="tree, text",
                height=image_height,
                width=image_width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                sync_weight=sync_weight,
                sync_decay_rate=sync_decay_rate,
                sync_freq=sync_freq,
                sync_thres=sync_thres,
                stride=stride,
                loop_closure=True,
                condition=True,
                inpaint_mask=mask_image,
                rendered_image=init_image,
                anchor_view_idx=anchor_view_idx,
            )

            # img.save(self.run_dir / f"{layer:02d}_synced_output.png")

            new_img = ToTensor()(img).unsqueeze(0).to(self.device)
            mask_image_ = mask_image.expand(-1, 3, -1, -1).bool()
            loss = (
                F.mse_loss(new_img[~mask_image_], init_image[~mask_image_]).cpu().item()
            )
            print(f"[INFO] Sky Loss: {loss}")

            # move conditioning image to the leftmost, and save it
            if layer == 0:
                new_img_ = torch.cat(
                    (new_img[:, :, :, w_start:], new_img[:, :, :, :w_start]), dim=-1
                )
                img = ToPILImage()(new_img_[0])
                img.save(self.run_dir / f"{layer:02d}_sky_leftmost.png")
            os.makedirs(f"./sky_image/{example_name}", exist_ok=True)
            img.save(f"./sky_image/{example_name}/sky_{layer}.png")
            imgs.append(img)
        img = linear_blend(imgs)
        # img.save(self.run_dir / f"sky_0_1_blend.png")

        image_height = img.size[-1]
        equatorial_radius = 0.02
        # range: FOV
        camera_angle_x = 2 * np.arctan(512 / (2 * self.init_focal_length))
        min_latitude = (
            -camera_angle_x / 2 - (image_height / 512 - 1) * camera_angle_x
        )  # Starting latitude of the band
        max_latitude = camera_angle_x / 2  # Ending latitude of the band

        latitude = torch.linspace(min_latitude, max_latitude, image_height)
        longitude_offset = (
            -camera_angle_x / 2
        )  # The conditioning img is the leftmost, we need to offset the longitude to let the 3D points align with the panorama
        longitude = torch.linspace(
            longitude_offset, longitude_offset + 2 * np.pi, image_width
        )

        lat, lon = torch.meshgrid(latitude, longitude, indexing="ij")

        # Pytorch3d coord system: +x: left, +y: up, +z: forward
        x = -equatorial_radius * torch.cos(lat) * torch.sin(lon)
        z = equatorial_radius * torch.cos(lat) * torch.cos(lon)
        y = -equatorial_radius * torch.sin(lat)

        points = torch.stack((x, y, z), -1)

        # Flatten the points for batch processing
        points_flat = points.reshape(-1, 3)

        # Assuming 'self.device' is the PyTorch device you want to use
        new_points_3d = points_flat.to(self.device)

        # img.save(self.run_dir / "sky.png")

        image_latest = ToTensor()(img).unsqueeze(0).to(self.device)
        colors = rearrange(image_latest, "b c h w -> (h w b) c")

        # Remove points below the ground height
        mask_above_ground = (
            new_points_3d[:, 1] >= -9999
        )  # set it to -9999 to use all points
        new_points_3d = new_points_3d[mask_above_ground]
        colors = colors[mask_above_ground]

        self.update_current_pc(new_points_3d, colors, gen_sky=True)
        # return

        # generate the upper part of the sky
        self.depth_latest[:] = self.sky_hard_depth
        self.disparity_latest[:] = 1.0 / self.sky_hard_depth
        self.depth_latest = self.depth_latest.to(self.device)
        self.disparity_latest = self.disparity_latest.to(self.device)

        ########## Generate downsampled points ############
        image_height_down, image_width_down = int(image_height / 2), int(
            image_width / 2
        )
        img_down = img.resize(
            (image_width_down, image_height_down), Image.Resampling.LANCZOS
        )
        latitude_down = torch.linspace(min_latitude, max_latitude, image_height_down)
        longitude_offset = (
            -camera_angle_x / 2
        )  # The conditioning img is the leftmost, we need to offset the longitude to let the 3D points align with the panorama
        longitude_down = torch.linspace(
            longitude_offset, longitude_offset + 2 * np.pi, image_width_down
        )

        lat_down, lon_down = torch.meshgrid(
            latitude_down, longitude_down, indexing="ij"
        )

        x_down = -equatorial_radius * torch.cos(lat_down) * torch.sin(lon_down)
        z_down = equatorial_radius * torch.cos(lat_down) * torch.cos(lon_down)
        y_down = -equatorial_radius * torch.sin(lat_down)

        points_down = torch.stack((x_down, y_down, z_down), -1)
        points_flat_down = points_down.reshape(-1, 3)
        new_points_3d_down = points_flat_down.to(self.device)

        image_latest_down = ToTensor()(img_down).unsqueeze(0).to(self.device)
        colors_down = rearrange(image_latest_down, "b c h w -> (h w b) c")

        # mask_above_ground = new_points_3d_down[:, 1] >= -0.0003
        if self.config['example_name'] in [
            'kite_1', 'kite_2', 'to_water_duck', 'tree_wind', "sand_house"
        ]:
            mask_above_ground = new_points_3d_down[:, 1] >= -9999 # set it to -9999 to use all points
        else:
            mask_above_ground = new_points_3d_down[:, 1] >= -0.0003
        new_points_3d_down = new_points_3d_down[mask_above_ground]
        colors_down = colors_down[mask_above_ground]
        self.sky_pc_downsampled = {"xyz": new_points_3d_down, "rgb": colors_down}

        self.generate_sky_cameras()
        print("No using sky top for efficiency.")
        return

    @torch.no_grad()
    def get_camera_by_js_view_matrix(self, view_matrix, xyz_scale=1.0):
        """
        args:
            view_matrix: list of 16 elements, representing the view matrix of the camera
            xyz_scale: This was used to scale the x, y, z coordinates of the camera when converting to 3DGS.
                Need to convert it back.
        return:
            camera: PyTorch3D camera object
        """
        view_matrix = torch.tensor(
            view_matrix, device=self.device, dtype=torch.float
        ).reshape(4, 4)
        xy_negate_matrix = torch.tensor(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            device=self.device,
            dtype=torch.float,
        )
        view_matrix_negate_xy = view_matrix @ xy_negate_matrix
        R = view_matrix_negate_xy[:3, :3].unsqueeze(0)
        T = view_matrix_negate_xy[3, :3].unsqueeze(0)
        camera = self.get_camera_at_origin()
        camera.R = R
        camera.T = T / xyz_scale
        return camera

    @torch.no_grad()
    def update_sky_mask(self):
        sky_mask_latest, sem_seg = self.generate_sky_mask(
            self.image_latest, return_sem_seg=True
        )
        self.sky_mask_latest = sky_mask_latest[None, None, :]
        return sem_seg

    @torch.no_grad()
    def generate_sky_mask(self, input_image=None, return_sem_seg=False):
        if input_image is not None:
            image = ToPILImage()(input_image.squeeze())
        else:
            image = ToPILImage()(self.image_latest.squeeze())

        segmenter_input = self.segment_processor(
            image, ["semantic"], return_tensors="pt"
        )
        segmenter_input = {
            name: tensor.to("cuda") for name, tensor in segmenter_input.items()
        }
        segment_output = self.segment_model(**segmenter_input)
        pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
            segment_output, target_sizes=[image.size[::-1]]
        )[0]
        sky_mask = pred_semantic_map == 2  # 2 for ade20k, 119 for coco
        if self.sky_erode_kernel_size > 0:
            sky_mask = (
                erosion(
                    sky_mask.float()[None, None],
                    kernel=torch.ones(
                        self.sky_erode_kernel_size, self.sky_erode_kernel_size
                    ).to(self.device),
                ).squeeze()
                > 0.5
            )
        if return_sem_seg:
            return sky_mask, pred_semantic_map
        else:
            return sky_mask
    
    @torch.no_grad()
    def generate_ground_mask_SAM(self, ground_ids, input_image=None):
        if input_image is not None:
            image_pil = ToPILImage()(input_image.squeeze())
        else:
            image_pil = ToPILImage()(self.image_latest.squeeze())
        image_np = np.array(image_pil)
        sam_masks = self.mask_generator.generate(image_np)
        sam_masks_np = []
        os.makedirs(self.run_dir / "segmentation_ground", exist_ok=True)
        for sid, sam_mask in enumerate(sam_masks):
            if sam_mask['area'] < 100:
                continue
            sam_masks_np.append(sam_mask['segmentation'])   # (512, 512) bool numpy array
            sam_mask = sam_mask['segmentation'] * 255
            sam_mask = sam_mask.astype(np.uint8)
            cv2.imwrite((self.run_dir / f"segmentation_ground/sam_mask_{sid:02d}.png").as_posix(), sam_mask)
            cv2.imwrite((self.run_dir / f"segmentation_ground/sam_mask_{sid:02d}_rgb.png").as_posix(), (sam_mask[:,:,None]/255).astype(np.uint8) * image_np[:,:,[2,1,0]])
        
        # assert len(ground_ids) > 0, 'requires valid ground_ids for SAM masks output'

        ground_mask = torch.zeros((512, 512), device=self.device, dtype=torch.bool)

        for idx in ground_ids:
            this_ground_mask = sam_masks[sid]['segmentation']
            this_ground_mask = torch.from_numpy(this_ground_mask).to(self.device)
            ground_mask = ground_mask | this_ground_mask
        
        if self.config["ground_erode_kernel_size"] > 0:
            ground_mask = (
                erosion(
                    ground_mask.float()[None, None],
                    kernel=torch.ones(
                        self.config["ground_erode_kernel_size"],
                        self.config["ground_erode_kernel_size"],
                    ).to(self.device),
                ).squeeze()
                > 0.5
            )
        
        return ground_mask
    
    @torch.no_grad()
    def generate_ground_mask(self, sem_map=None, input_image=None):
        if sem_map is None:
            if input_image is not None:
                image = ToPILImage()(input_image.squeeze())
            else:
                image = ToPILImage()(self.image_latest.squeeze())

            segmenter_input = self.segment_processor(
                image, ["semantic"], return_tensors="pt"
            )
            segmenter_input = {
                name: tensor.to("cuda") for name, tensor in segmenter_input.items()
            }
            segment_output = self.segment_model(**segmenter_input)
            pred_semantic_map = (
                self.segment_processor.post_process_semantic_segmentation(
                    segment_output, target_sizes=[image.size[::-1]]
                )[0]
            )
            sem_map = pred_semantic_map
        # 3: floor; 6: road; 9: grass; 11: pavement; 13: earth; 26: sea; 29: field; 46: sand; 128: lake
        ground_mask = (
            (sem_map == 3)
            | (sem_map == 6)
            | (sem_map == 9)
            | (sem_map == 11)
            | (sem_map == 13)
            | (sem_map == 26)
            | (sem_map == 29)
            | (sem_map == 46)
            | (sem_map == 128)
        )
        if self.config["ground_erode_kernel_size"] > 0:
            ground_mask = (
                erosion(
                    ground_mask.float()[None, None],
                    kernel=torch.ones(
                        self.config["ground_erode_kernel_size"],
                        self.config["ground_erode_kernel_size"],
                    ).to(self.device),
                ).squeeze()
                > 0.5
            )
        return ground_mask

    @torch.no_grad()
    def generate_grad_magnitude(self, disparity):
        vmin, vmax = disparity.min(), disparity.max()
        normalized_disparity = (disparity - vmin) / (vmax - vmin)
        cmap = plt.get_cmap("viridis")
        rgb_image = cmap(normalized_disparity)
        rgb_image = rgb_image[..., 1]
        disparity = np.uint8(rgb_image * 255)

        ToPILImage()(disparity).save(
            self.run_dir
            / "images"
            / "disparity_gradient"
            / f"{self.kf_idx}_normalized_disparity.png"
        )

        # Compute gradients along the x and y axis
        grad_x = cv2.Sobel(disparity, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(disparity, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        grad_magnitude = cv2.magnitude(grad_x, grad_y)
        grad_magnitude = cv2.normalize(
            grad_magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        threshold = 50
        mask = torch.from_numpy(grad_magnitude > threshold)
        return mask

    @torch.no_grad()
    def generate_layer(
        self,
        pred_semantic_map=None,
        ground_ids=["3", "6", "9", "11", "13", "26", "29", "46", "52", "128"],
        foreground_ids=[4, 76, 83, 87],
    ):
        # if not isinstance(ground_ids[0], str):
        #     ground_ids = [str(id) for id in ground_ids]
        # if not isinstance(foreground_ids[0], str):
        #     foreground_ids = [str(id) for id in foreground_ids]

        self.image_latest_init = copy.deepcopy(self.image_latest)
        self.depth_latest_init = copy.deepcopy(self.depth_latest)
        self.disparity_latest_init = copy.deepcopy(self.disparity_latest)
        if pred_semantic_map is None:
            image = ToPILImage()(self.image_latest.squeeze())

            segmenter_input = self.segment_processor(
                image, ["semantic"], return_tensors="pt"
            )
            segmenter_input = {
                name: tensor.to("cuda") for name, tensor in segmenter_input.items()
            }
            segment_output = self.segment_model(**segmenter_input)
            pred_semantic_map = (
                self.segment_processor.post_process_semantic_segmentation(
                    segment_output, target_sizes=[image.size[::-1]]
                )[0]
            )

        unique_elements = torch.unique(pred_semantic_map)
        masks = {
            str(element.item()): (pred_semantic_map == element)
            for element in unique_elements
        }

        # erosion the mask to avoid margin effect
        disparity_np = self.disparity_latest.squeeze().cpu().numpy()
        grad_magnitude_mask = self.generate_grad_magnitude(disparity_np)
        mask_disocclusion = np.full((512, 512), False, dtype=bool)

        dilation_kernel = torch.ones(9, 9).to(self.device)

        # also use SAM to generate the masks candiate
        image_pil = ToPILImage()(self.image_latest.squeeze())
        image_np = np.array(image_pil)
        sam_masks = self.mask_generator.generate(image_np)
        sam_masks_np = []
        for sid, sam_mask in enumerate(sam_masks):
            if sam_mask['area'] < 100:
                continue
            sam_masks_np.append(sam_mask['segmentation'])   # (512, 512) bool numpy array
            sam_mask = sam_mask['segmentation'] * 255
            sam_mask = sam_mask.astype(np.uint8)
            cv2.imwrite((self.run_dir / f"segmentation/sam_mask_{sid:02d}.png").as_posix(), sam_mask)
            cv2.imwrite((self.run_dir / f"segmentation/sam_mask_{sid:02d}_rgb.png").as_posix(), (sam_mask[:,:,None]/255).astype(np.uint8) * image_np[:,:,[2,1,0]])

        for id, mask in masks.items():
            # exclude 3: floor; 6: road; 9: grass; 11: pavement; 13: earth; 26: sea; 29: field; 46: sand; 52: path, 128: lake
            if id in ground_ids:
                continue
                continue
            # -- 1. Dilate each segment --#
            mask = (
                dilation((mask).float()[None, None], kernel=dilation_kernel)
                .squeeze()
                .cpu()
                > 0.5
            )
            # 4: tree; 6: boat; 83: truck; 87: street lamp
            if id in foreground_ids:
                mask_disocclusion |= mask.numpy()
                continue
            labeled_array, num_features = label(mask)
            for i in range(1, num_features + 1):
                # -- 2. Fetch all disparity values, within the segment --#
                mask_i = labeled_array == i
                disp_pixels = disparity_np[mask_i]
                disparity_mean = disp_pixels.mean()
                # -- 3. [Remove distant segments] --#
                if disparity_mean < np.percentile(disparity_np, 60):
                    continue
                # -- 4. [Find the disparity boundary] --#
                grad_magnitude_segment = grad_magnitude_mask[mask_i]
                # -- 5. [Remove segments without disparity boundaries] --#
                if grad_magnitude_segment.float().mean() < 0.02:
                    continue
                # -- 6. [Find boundary pixels] --#
                segment_boundary = np.where(mask_i, grad_magnitude_mask, 0)
                if disparity_np[segment_boundary != 0].mean() > np.percentile(
                    disp_pixels, 70
                ):
                    continue
                # -- 7. [Find big-enough region] --#
                if mask_i.mean() < 0.001:
                    continue
                # -- 8. [Find non-road region] --#
                # ToPILImage()((self.image_latest.cpu() * mask_i)[0]).save('test.png')
                mask_i_erosion = (
                    erosion(
                        torch.from_numpy(mask_i).float()[None, None],
                        kernel=dilation_kernel.cpu(),
                    ).squeeze()
                    > 0.5
                )
                disp_pixels = disparity_np[mask_i_erosion]
                p20 = np.percentile(disp_pixels, 20)
                p80 = np.percentile(disp_pixels, 80)
                if (
                    1 / p20 - 1 / p80 > 0.0003 and mask_i.mean() > 0.05
                ):  # indicates it is a road
                    continue

                save_prompt = False
                # print(i, "disparity_mean:", disparity_mean, "segment_disparity_mean:", disparity_np[segment_boundary!=0].mean())
                mask_disocclusion |= mask_i

        '''
        This is the current option to get the best object masks:
        You need to set in config the object_split_mask_sam_ids, and the order matters!
        then directly use those masks as object masks, no erosion or dilation, and rewrite the mask_disocclusion
        '''
        if "object_split_mask_sam_ids" not in self.config.keys():
            print("No object_split_mask_sam_ids in config, check the segmentation folder to select the object masks ids!!")
            raise NotImplementedError
        else:
            object_split_mask_sam_ids = self.config["object_split_mask_sam_ids"]
            mask_disocclusion = np.full((512, 512), False, dtype=bool)
            self.object_masks = []
            for object_number_id, sid in enumerate(object_split_mask_sam_ids):
                combined_mask = np.full((512, 512), False, dtype=bool)
                try:
                    len_list = len(sid)
                    for siid in sid:
                        mask_disocclusion |= sam_masks_np[siid]
                        combined_mask |= sam_masks_np[siid]
                    # if the len(sid) == 0, you can pass this and should have an existing mask in tmp/
                except:
                    mask_disocclusion |= sam_masks_np[sid]
                    combined_mask |= sam_masks_np[sid]
                
                # also if in tmp/, there is {example_name}_object_{id}.png, then also use it
                if os.path.exists(f"tmp/{self.config['example_name']}_object_{object_number_id}.png"):
                    existing_sam_mask = np.array(Image.open(f"tmp/{self.config['example_name']}_object_{object_number_id}.png"))
                    existing_sam_mask = existing_sam_mask > 0
                    combined_mask |= existing_sam_mask
                    mask_disocclusion |= existing_sam_mask

                self.object_masks.append(torch.from_numpy(combined_mask).float().unsqueeze(0).unsqueeze(0).to(self.device))
        cv2.imwrite((self.run_dir / f"segmentation/sam_mask_disocclusion.png").as_posix(), (mask_disocclusion*255).astype(np.uint8))
        
        inpainting_prompt = self.config["base_inpainting_prompt"]
        inpainting_negative_prompt = self.config["base_inpainting_negative_prompt"]
        print("Base layer inpainting_prompt: ", inpainting_prompt)
        print("Base layer inpainting_negative_prompt: ", inpainting_negative_prompt)
        mask_disocclusion = torch.from_numpy(mask_disocclusion)[None, None]

        """Outside of this function, mask_disocclusion will be used to update point cloud and compute depth; 
        # For depth, we want the mask to be accurate, because we want to align correctly;
        # For point cloud, we also want the mask accurate, because we don't want to attach sky edges to the trees."""

        self.mask_disocclusion = erosion(
            mask_disocclusion.float().to(self.device), kernel=dilation_kernel
        )

        is_tmp = False  # True
        is_tmp = True if self.config['example_name'] in [
            'venice_1',
            'cake',
            'smoke_can',
            'cloth_hang_two',
            'domino',
            'jam',
            'boat_river_1',
            'boat_river_2',
            'boat_river_3',
            'boat_river_4',
            'kite_1',
            'kite_2',
            'food_wine_1',
            'food_wine_2',
            'sand_castle',
            'sand_house',
            'three_macaron',
            'three_persimmon',
            'snow_statue',
            'snow_man',
            'to_water_duck',
            'smoke_interaction_1',
            'smoke_interaction_2',
            'flower_wind',
            'cloth_fruit',
            'tree_wind',
            'demo_1',
            'demo_3',
            'demo_4',
            'demo_6',
            'demo_7',
            'demo_8',
            'demo_9',
        ] else False

        if is_tmp:
            if os.path.exists(f"tmp/{self.config['example_name']}_object_mask.png"):
                mask_disocclusion = ToTensor()(
                    Image.open(f"tmp/{self.config['example_name']}_object_mask.png")
                    .convert("L")
                    .resize((512, 512))
                )[None].bool()
                self.mask_disocclusion = mask_disocclusion.float().to(self.device)
        
        inpaint_mask = (
            self.mask_disocclusion > 0.5
        )  # Erode a bit to prevent over-inpaint
        self.inpaint(
            self.image_latest_init,
            inpaint_mask=inpaint_mask,
            inpainting_prompt=inpainting_prompt,
            negative_prompt=inpainting_negative_prompt,
            mask_strategy=np.max,
            diffusion_steps=50,
        )
        inpainter_output = self.image_latest

        stitch_mask = dilation(
            mask_disocclusion.float().to(self.device),
            kernel=torch.ones(5, 5).to(self.device),
        )  # keep it slightly dilated to prevent dirty artifacts
        
        if is_tmp:
            if os.path.exists(f"tmp/{self.config['example_name']}_base_layer.png"):
                inpainter_output = ToTensor()(
                    Image.open(f"tmp/{self.config['example_name']}_base_layer.png").resize((512, 512))
                )[None].to(self.device)
                stitch_mask = dilation(
                    mask_disocclusion.float().to(self.device),
                    kernel=torch.ones(7, 7).to(self.device),
                )  # keep it slightly dilated to prevent dirty artifacts
        self.image_latest = soft_stitching(
            inpainter_output, self.image_latest_init, stitch_mask, sigma=1, blur_size=3
        )
        # ToPILImage()(grad_magnitude_mask.float()).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_grad_magnitude_mask.png')
        # ToPILImage()((self.image_latest.cpu() * mask_disocclusion.float())[0]).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_mask_disocclusion.png')
        # ToPILImage()((self.image_latest_init.cpu() * inpaint_mask.float())[0]).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_inpaint_mask.png')
        # ToPILImage()(self.image_latest_init[0]).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_image_init.png')
        # ToPILImage()(self.image_latest[0]).save(self.run_dir / 'images' / 'layer' / f'{self.kf_idx:02d}_remove_disocclusion.png')

    @torch.no_grad()
    def transform_all_cam_to_current_cam(self, center=False):
        """Transform all self.cameras such that the current camera is at the origin."""

        if self.cameras != []:
            if not center:
                inv_current_camera_RT = (
                    self.cameras[-1]
                    .get_world_to_view_transform()
                    .inverse()
                    .get_matrix()
                )
            else:
                inv_current_camera_RT = (
                    self.cameras[self.center_camera_idx]
                    .get_world_to_view_transform()
                    .inverse()
                    .get_matrix()
                )

            for cam in self.cameras:
                cam_RT = cam.get_world_to_view_transform().get_matrix()
                new_cam_RT = inv_current_camera_RT @ cam_RT
                cam.R = new_cam_RT[:, :3, :3]
                cam.T = new_cam_RT[:, 3, :3]

    @torch.no_grad()
    def set_current_camera(self, camera, archive_camera=False):
        self.current_camera = camera
        if archive_camera:
            self.cameras_archive.append(copy.deepcopy(camera))

    @torch.no_grad()
    def set_cameras(self, rotation_path):
        move_left_count = 0
        move_right_count = 0
        for rotation in rotation_path:
            new_camera = copy.deepcopy(self.cameras[-1])

            if rotation == 0:
                forward_speed_multiplier = -1.0
                right_multiplier = 0
                camera_speed = self.camera_speed

                # If the camera is not centered, rotate the camera by rotation matrix
                if move_left_count != 0 or move_right_count != 0:
                    # moving backward and previous motion is moving right/left
                    new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                    move_left_count = 0
                    move_right_count = 0

            elif abs(rotation) == 2:
                # If the camera is not centered, rotate the camera by rotation matrix
                if rotation > 0:
                    move_left_count += 1
                    # moving left and previous motion is moving right
                    if move_right_count != 0:
                        new_camera = copy.deepcopy(
                            self.cameras[self.scene_cameras_idx[-1]]
                        )
                        move_right_count = 0
                else:
                    move_right_count += 1
                    # moving right and previous motion is moving left
                    if move_left_count != 0:
                        new_camera = copy.deepcopy(
                            self.cameras[self.scene_cameras_idx[-1]]
                        )
                        move_left_count = 0

                forward_speed_multiplier = 0
                right_multiplier = 0
                camera_speed = 0
                theta = torch.tensor(self.rotation_range_theta * rotation / 2)
                rotation_matrix = torch.tensor(
                    [
                        [torch.cos(theta), 0, torch.sin(theta)],
                        [0, 1, 0],
                        [-torch.sin(theta), 0, torch.cos(theta)],
                    ],
                    device=self.device,
                )
                new_camera.R[0] = rotation_matrix @ new_camera.R[0]

            elif (
                abs(rotation) == 1
            ):  # Pre-compute camera movement, accounting for a set of kfinterp rotations
                # If the camera is not centered, rotate the camera by rotation matrix
                if move_left_count != 0 or move_right_count != 0:
                    # moving backward and previous motion is moving right/left
                    new_camera = copy.deepcopy(self.cameras[self.scene_cameras_idx[-1]])
                    move_left_count = 0
                    move_right_count = 0

                theta_frame = (
                    torch.tensor(self.rotation_range_theta / (self.interp_frames + 1))
                    * rotation
                )
                sin = torch.sum(
                    torch.stack(
                        [
                            torch.sin(i * theta_frame)
                            for i in range(1, self.interp_frames + 2)
                        ]
                    )
                )
                cos = torch.sum(
                    torch.stack(
                        [
                            torch.cos(i * theta_frame)
                            for i in range(1, self.interp_frames + 2)
                        ]
                    )
                )
                forward_speed_multiplier = -1.0 / (self.interp_frames + 1) * cos.item()
                right_multiplier = -1.0 / (self.interp_frames + 1) * sin.item()
                camera_speed = self.camera_speed * self.camera_speed_multiplier_rotation

                theta = torch.tensor(self.rotation_range_theta * rotation)
                rotation_matrix = torch.tensor(
                    [
                        [torch.cos(theta), 0, torch.sin(theta)],
                        [0, 1, 0],
                        [-torch.sin(theta), 0, torch.cos(theta)],
                    ],
                    device=self.device,
                )
                new_camera.R[0] = rotation_matrix @ new_camera.R[0]

            elif rotation == 3:
                continue

            move_dir = torch.tensor(
                [[-right_multiplier, 0.0, -forward_speed_multiplier]],
                device=self.device,
            )

            # move camera backwards
            new_camera.T += camera_speed * move_dir
            self.cameras.append(copy.deepcopy(new_camera))

        return new_camera

    @torch.no_grad()
    def generate_cameras(self, rotation_path):
        print("-- generating 360-degree cameras...")
        # Generate init camera for each scene
        camera = self.get_camera_at_origin()
        self.cameras.append(copy.deepcopy(camera))
        self.scene_cameras_idx.append(len(self.cameras) - 1)
        self.transform_all_cam_to_current_cam()
        # Generate camera sequence based on rotation_path
        self.set_cameras(rotation_path)
        self.center_camera_idx = 0
        self.transform_all_cam_to_current_cam(True)
        print("-- generated 360-degree cameras!")

    @torch.no_grad()
    def generate_sky_cameras(self):
        print("-- generating sky cameras...")
        cameras_cache = copy.deepcopy(self.cameras)
        init_len = len(self.cameras)

        # Generate cameras for sky generation
        for i in tqdm(range(1)):
            delta = -torch.tensor(torch.pi) / (8) * (i + 1)
            for camera_id in range(init_len):
                self.center_camera_idx = camera_id
                self.transform_all_cam_to_current_cam(True)
                new_camera = copy.deepcopy(self.cameras[camera_id])

                rotation_matrix = torch.tensor(
                    [
                        [1, 0, 0],
                        [0, torch.cos(delta), -torch.sin(delta)],
                        [0, torch.sin(delta), torch.cos(delta)],
                    ],
                    device=self.device,
                )
                new_camera.R[0] = rotation_matrix @ new_camera.R[0]

                self.cameras.append(copy.deepcopy(new_camera))
        self.center_camera_idx = 0
        self.transform_all_cam_to_current_cam(True)
        self.sky_cameras = copy.deepcopy(self.cameras)
        self.cameras = cameras_cache
        print("-- generated sky cameras!")

    @torch.no_grad()
    def set_kf_param(
        self, inpainting_resolution, inpainting_prompt, adaptive_negative_prompt
    ):
        super().set_frame_param(
            inpainting_resolution=inpainting_resolution,
            inpainting_prompt=inpainting_prompt,
            adaptive_negative_prompt=adaptive_negative_prompt,
        )

    @torch.no_grad()
    def refine_disp_with_segments(
        self,
        save_intermediates=False,
        keep_threshold_disp_range=10,
        no_refine_mask=None,
        existing_mask=None,
        existing_disp=None,
    ):
        """
        args:
            no_refine_mask: basically it is ground mask, if not None. Then, if a SAM segment has significant intersection with the ground, then we discard it.
            existing_mask: if not None, then we are refining layer-inpainted disp. In this case, most regions have existing fixed disp.
                            For a SAM segment, if it intersects with existing_mask significantly,
                            then we should only use the values from existing_disp, not from the current estimate, to minimize the gaps.
        """
        print("Refining disparity with segments...")
        if save_intermediates:
            (self.run_dir / "refine_intermediates").mkdir(parents=True, exist_ok=True)
        image = ToPILImage()(self.image_latest.squeeze())
        image_np = np.array(image)
        masks = self.mask_generator.generate(image_np)
        sorted_mask = sorted(
            masks, key=(lambda x: x["area"]), reverse=False
        )  # Iterate from small to large, finally large will have higher priority.
        min_mask_area = 100
        sorted_mask = [m for m in sorted_mask if m["area"] > min_mask_area]

        if save_intermediates:
            save_sam_anns(
                masks,
                self.run_dir / "refine_intermediates" / f"kf{self.kf_idx:02}_SAM.png",
            )

        save_sam_anns(masks, "tmp/sam.png")

        disparity_np = self.disparity_latest.squeeze().cpu().numpy()

        refined_disparity = refine_disp_with_segments_2(
            disparity_np,
            sorted_mask,
            keep_threshold=keep_threshold_disp_range,
            no_refine_mask=no_refine_mask,
            existing_mask=existing_mask,
            existing_disp=existing_disp,
        )

        if save_intermediates:
            save_depth_map(
                1 / refined_disparity,
                self.run_dir / "refine_intermediates" / f"kf{self.kf_idx:02}_p1_SAM",
            )

        refined_depth = 1 / refined_disparity

        refined_depth = torch.from_numpy(refined_depth).to(self.device)
        refined_disparity = torch.from_numpy(refined_disparity).to(self.device)

        self.depth_latest[0, 0] = refined_depth
        self.disparity_latest[0, 0] = refined_disparity

        print("Refining done!")
        return refined_depth, refined_disparity

    @torch.no_grad()
    def generate_visible_pc(self):
        camera = self.current_camera
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius=0.003,
            points_per_pixel=8,
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=camera, raster_settings=raster_settings
            ),
            compositor=SoftmaxImportanceCompositor(
                background_color=BG_COLOR, softmax_scale=1.0
            ),
        )
        points, colors = self.get_combined_pc()["xyz"], self.get_combined_pc()["rgb"]
        point_cloud = Pointclouds(points=[points], features=[colors])
        images, fragment_idx = renderer(point_cloud, return_fragment_idx=True)
        fragment_idx = fragment_idx[..., :1]

        n_kf1_points = points.shape[0]
        fragment_idx = fragment_idx.reshape(-1)
        visible_points_idx = (fragment_idx < n_kf1_points) & (fragment_idx >= 0)
        fragment_idx = fragment_idx[visible_points_idx]

        if self.current_visible_pc is None:
            self.current_visible_pc = {
                "xyz": points[fragment_idx],
                "rgb": colors[fragment_idx],
            }
        else:
            self.current_visible_pc = {
                "xyz": torch.cat(
                    [self.current_visible_pc["xyz"], points[fragment_idx]], dim=0
                ),
                "rgb": torch.cat(
                    [self.current_visible_pc["rgb"], colors[fragment_idx]], dim=0
                ),
            }

    @torch.no_grad()
    def render(
        self,
        archive_output=False,
        camera=None,
        render_visible=False,
        render_sky=False,
        big_view=False,
        render_fg=False,
    ):
        camera = self.current_camera if camera is None else camera
        raster_settings = PointsRasterizationSettings(
            image_size=1536 if big_view else 512,
            radius=0.003,
            points_per_pixel=8,
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=camera, raster_settings=raster_settings
            ),
            compositor=SoftmaxImportanceCompositor(
                background_color=BG_COLOR, softmax_scale=1.0
            ),
        )
        if render_visible:
            points, colors = (
                self.current_visible_pc["xyz"],
                self.current_visible_pc["rgb"],
            )
        elif render_sky:
            points, colors = self.current_pc_sky["xyz"], self.current_pc_sky["rgb"]
        elif render_fg:
            points, colors = self.current_pc["xyz"], self.current_pc["rgb"]
        else:
            points, colors = (
                self.get_combined_pc()["xyz"],
                self.get_combined_pc()["rgb"],
            )

        point_cloud = Pointclouds(points=[points], features=[colors])
        images, zbuf, bg_mask = renderer(
            point_cloud, return_z=True, return_bg_mask=True
        )

        rendered_image = rearrange(images, "b h w c -> b c h w")
        inpaint_mask = bg_mask.float()[:, None, ...]
        rendered_depth = rearrange(zbuf[..., 0:1], "b h w c -> b c h w")
        rendered_depth[rendered_depth < 0] = 0

        if archive_output:
            self.rendered_image_latest = rendered_image
            self.rendered_depth_latest = rendered_depth
            self.mask_latest = inpaint_mask

        return {
            "rendered_image": rendered_image,
            "rendered_depth": rendered_depth,
            "inpaint_mask": inpaint_mask,
        }

    @torch.no_grad()
    def archive_latest(self, idx=None):
        if idx is None:
            idx = self.kf_idx
        vmax = 0.006
        super().archive_latest(idx=idx, vmax=vmax)
        self.rendered_images.append(self.rendered_image_latest)
        self.rendered_depths.append(self.rendered_depth_latest)
        self.sky_mask_list.append(~self.sky_mask_latest.bool())

        # save_root = Path(self.run_dir) / "images"
        # save_root.mkdir(exist_ok=True, parents=True)
        # (save_root / "rendered_images").mkdir(exist_ok=True, parents=True)
        # (save_root / "rendered_depths").mkdir(exist_ok=True, parents=True)

        # ToPILImage()(self.rendered_image_latest[0]).save(save_root / "rendered_images" / f"{idx:03d}.png")
        # save_depth_map(self.rendered_depth_latest.clamp(0).cpu().numpy(), save_root / "rendered_depths" / f"{idx:03d}.png", vmax=vmax)


def get_extrinsics(camera):
    extrinsics = torch.cat([camera.R[0], camera.T.T], dim=1)
    padding = torch.tensor([[0, 0, 0, 1]], device=extrinsics.device)
    extrinsics = torch.cat([extrinsics, padding], dim=0)
    return extrinsics


def save_point_cloud_as_ply(points, filename="output.ply", colors=None):
    """
    Save a PyTorch tensor of shape [N, 3] as a PLY file. Optionally with colors.

    Parameters:
    - points (torch.Tensor): The point cloud tensor of shape [N, 3].
    - filename (str): The name of the output PLY file.
    - colors (torch.Tensor, optional): The color tensor of shape [N, 3] with values in [0, 1]. Default is None.
    """

    assert (
        points.dim() == 2 and points.size(1) == 3
    ), "Input tensor should be of shape [N, 3]."

    if colors is not None:
        assert (
            colors.dim() == 2 and colors.size(1) == 3
        ), "Color tensor should be of shape [N, 3]."
        assert points.size(0) == colors.size(
            0
        ), "Points and colors tensors should have the same number of entries."

    # Header for the PLY file
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.size(0)}",
        "property float x",
        "property float y",
        "property float z",
    ]

    # Add color properties to header if colors are provided
    if colors is not None:
        header.extend(
            ["property uchar red", "property uchar green", "property uchar blue"]
        )

    header.append("end_header")

    # Write to file
    with open(filename, "w") as f:
        for line in header:
            f.write(line + "\n")

        for i in range(points.size(0)):
            line = f"{points[i, 0].item()} {points[i, 1].item()} {points[i, 2].item()}"

            # Add color data to the line if colors are provided
            if colors is not None:
                # Scale color values from [0, 1] to [0, 255] and convert to integers
                r, g, b = (colors[i] * 255).clamp(0, 255).int().tolist()
                line += f" {r} {g} {b}"

            f.write(line + "\n")


def convert_pytorch3d_kornia(
    camera, focal_length, size=512, update_intrinsics_parameters=None, new_size=512
):
    transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
    transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)

    pt3d_to_kornia = torch.diag(torch.tensor([-1.0, -1, 1, 1], device=camera.device))
    transform_matrix_w2c_kornia = pt3d_to_kornia @ transform_matrix_w2c_pt3d

    extrinsics = transform_matrix_w2c_kornia.unsqueeze(0)
    h = torch.tensor([size], device="cuda")
    w = torch.tensor([size], device="cuda")
    K = torch.eye(4)[None].to("cuda")
    K[0, 0, 2] = size // 2
    K[0, 1, 2] = size // 2
    K[0, 0, 0] = focal_length
    K[0, 1, 1] = focal_length
    if update_intrinsics_parameters is not None:
        u0, v0, w_crop, h_crop, p_left, p_right, p_up, p_down, scale = (
            update_intrinsics_parameters
        )
        new_cx = (K[0, 0, 2] - u0 + p_left) * scale
        new_cy = (K[0, 1, 2] - v0 + p_up) * scale
        new_fx = K[0, 0, 0] * scale
        new_fy = K[0, 1, 1] * scale
        K[0, 0, 2] = new_cx
        K[0, 1, 2] = new_cy
        K[0, 0, 0] = new_fx
        K[0, 1, 1] = new_fy
        new_h = torch.tensor([new_size], device="cuda")
        new_w = torch.tensor([new_size], device="cuda")
        return PinholeCamera(K, extrinsics, new_h, new_w)

    return PinholeCamera(K, extrinsics, h, w)

    # transform_matrix_w2c_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
    # transform_matrix_w2c_pt3d = transform_matrix_w2c_pt3d.transpose(0, 1)

    # transform_matrix_c2w_pt3d = transform_matrix_w2c_pt3d.inverse()
    # pt3d_to_kornia = torch.diag(torch.tensor([-1, 1, -1, 1], device=self.device))
    # transform_matrix_w2c_kornia = transform_matrix_w2c_kornia @ pt3d_to_kornia

    # R = torch.clone(camera.R)
    # T = torch.clone(camera.T)
    # T[0, 0] = -T[0, 0]

    # extrinsics = torch.eye(4, device=R.device).unsqueeze(0)
    # extrinsics[:, :3, :3] = R
    # extrinsics[:, :3, 3] = T
    # h = torch.tensor([size], device="cuda")
    # w = torch.tensor([size], device="cuda")
    # K = torch.eye(4)[None].to("cuda")
    # K[0, 0, 2] = size // 2
    # K[0, 1, 2] = size // 2
    # K[0, 0, 0] = focal_length
    # K[0, 1, 1] = focal_length
    # return PinholeCamera(K, extrinsics, h, w)


def inpaint_cv2(rendered_image, mask_diff):
    """
    Performs inpainting on a single image using a corresponding mask, both provided as PyTorch tensors.

    Args:
    - rendered_image (torch.Tensor): A tensor of shape [batch_size, channels, height, width].
      This function uses only the first image in the batch for inpainting.
    - mask_diff (torch.Tensor): A tensor of shape [batch_size, 1, height, width] representing the inpainting mask.
      This function uses only the first mask in the batch.

    Returns:
    - torch.Tensor: Inpainted image in a tensor of shape [1, channels, height, width], where 'channels'
      corresponds to the same number of channels as the input image, typically 3 (RGB).

    """
    image_cv2 = rendered_image[0].permute(1, 2, 0).cpu().numpy()
    image_cv2 = (image_cv2 * 255).astype(np.uint8)
    mask_cv2 = mask_diff[0, 0].cpu().numpy()
    mask_cv2 = (mask_cv2 * 255).astype(np.uint8)
    inpainting = cv2.inpaint(image_cv2, mask_cv2, 3, cv2.INPAINT_TELEA)
    inpainting = torch.from_numpy(inpainting).permute(2, 0, 1).float() / 255
    return inpainting.unsqueeze(0)
