"""
Gaussian scene segmentation module for separating foreground and background.

This module separates the foreground and background of a reconstructed Gaussian scene 
to prepare for subsequent physics simulation. It generates segmented Gaussian point 
clouds and RGB-D images from different camera viewpoints.

Output:
    Running this script generates the following directories under 
    3d_result/{scene_name}/stage1_reconstruction/:
    - background/: Directory containing background Gaussian splats
    - foreground/: Directory containing foreground Gaussian splats
    - rgbd/: Directory containing RGB-D images and videos rendered from different viewpoints
"""

import torch
import os
import json
from pathlib import Path
import numpy as np
import imageio
from tqdm import tqdm
import yaml
import sys
from gsplat.rendering import rasterization_2dgs
sys.path.append("submodules/segment_anything")
from sam2.build_sam import build_sam2_video_predictor
from seg_video import generate_masks


def generate_circular_rgbd_from_foreground(foreground_splats, cam_pose_folder, masks_folder=None, output_dir="img_and_depth", scene_info_path=None, args=None):
    """
    Generate circular RGB-D images around the object from foreground Gaussian splats and apply masks
    
    Args:
        foreground_splats: Dictionary containing foreground Gaussian splat data
        cam_pose_folder: Path to camera pose folder
        masks_folder: Path to masks folder, if provided masks will be applied to RGB and depth images
        output_dir: Output folder path (default: "img_and_depth")
        scene_info_path: Path to scene info file for loading scene scale
    
    Returns:
        tuple: (rgb_images, depth_images, poses, intrinsics)
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masked_rgb"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masked_depth_img"), exist_ok=True)
    
    # Load scene scale information and camera ID information
    scene_scale = 1.0
    camera_ids = []
    if scene_info_path and os.path.exists(scene_info_path):
        with open(scene_info_path, 'r') as f:
            scene_info = json.load(f)
        scene_scale = scene_info.get('final_scale', 1.0)
        camera_ids = scene_info.get('camera_ids', [])
    else:
        print("Warning: No scene info file found, using default scale 1.0")
    
    # Load camera poses and intrinsics
    poses = []
    intrinsics = []
    num_views = len(os.listdir(cam_pose_folder))
    
    for i in range(num_views):
        pose_data = np.load(os.path.join(cam_pose_folder, f"camera_{i:04d}.npz"))
        # Assume file contains K and camtoworld
        K = pose_data['K']  # Camera intrinsic matrix [3, 3]
        camtoworld = pose_data['camtoworld']  # Camera extrinsic matrix [4, 4]
        
        # Apply scene scale correction to camera poses
        # camtoworld[:3, 3] /= scene_scale
        
        poses.append(camtoworld)
        intrinsics.append(K)
    # camera_path = os.path.join(args.save_dir, "3d", "camera_paths", "default.json")
    # from toolkit.camera import load_cameras_from_path
    # poses, intrinsics = load_cameras_from_path(camera_path)

    
    print(f"Applied scene scale {scene_scale} to camera poses")
    print(f"Loaded {len(poses)} camera poses")

    
    # Get image size (inferred from intrinsic matrix)
    K = intrinsics[0]
    width = 1280  # cx * 2
    height = 704  # cy * 2
    image_size = (width, height)
    
    # Render RGB-D images
    rgb_images = []
    depth_images = []
    
    print("Preprocessing Gaussian splat data...")
    means = foreground_splats['means'].cpu().numpy()
    quats = foreground_splats['quats'].cpu().numpy()
    scales = np.exp(foreground_splats['scales'].cpu().numpy())
    opacities = 1.0 / (1.0 + np.exp(-foreground_splats['opacities'].cpu().numpy()))

    # Process spherical harmonics coefficients

    sh0 = foreground_splats['sh0'].cpu().numpy()
    shN = foreground_splats['shN'].cpu().numpy()
    colors = np.concatenate([sh0, shN], axis=1)

    # Convert to torch tensors and move to GPU (only once)
    means_tensor = torch.from_numpy(means).float().cuda()
    quats_tensor = torch.from_numpy(quats).float().cuda()
    scales_tensor = torch.from_numpy(scales).float().cuda()
    opacities_tensor = torch.from_numpy(opacities).float().cuda()
    colors_tensor = torch.from_numpy(colors).float().cuda()
    
    print("Rendering RGB-D images...")
    for i, (pose, intrinsic) in enumerate(tqdm(zip(poses, intrinsics), desc="Rendering progress")):
        # Only convert camera parameters
        viewmat = np.linalg.inv(pose)
        viewmat_tensor = torch.from_numpy(viewmat).float().cuda().unsqueeze(0)
        intrinsic_tensor = torch.from_numpy(intrinsic).float().cuda().unsqueeze(0)
        
        # Render RGB and depth - reference utils/Scene_3d.py processing
        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            info,
        ) = rasterization_2dgs(
            means=means_tensor,
            quats=quats_tensor,
            scales=scales_tensor,
            opacities=opacities_tensor,
            colors=colors_tensor,
            viewmats=viewmat_tensor,
            Ks=intrinsic_tensor,
            width=width,
            height=height,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            sh_degree=0,
            render_mode="RGB+ED",
            near_plane=0.2,
            far_plane=200.0,
            radius_clip=0.0,
        )
        
        # Extract RGB and depth
        rgb = render_colors[0, ..., :3].cpu().detach().numpy()  # [H, W, 3]
        depth = render_median[0, ...].cpu().detach().numpy()    # May be [H, W] or [H, W, 1]
        
        # Ensure depth is 2D
        if depth.ndim == 3:
            depth = depth.squeeze(-1)  # Remove last dimension, from [H, W, 1] to [H, W]
        
        rgb_images.append(rgb)
        depth_images.append(depth)
        
        # Save images
        rgb_path = os.path.join(output_dir, "rgb", f"rgb_{i:04d}.png")

        # Save RGB image
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        imageio.imwrite(rgb_path, rgb_uint8)
        
        # Save depth image
        # np.save(depth_path, depth)
    sam2_predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", "submodules/sam2/checkpoints/sam2.1_hiera_large.pt")
    rgb_dir = os.path.join(output_dir, "rgb")
    masks = generate_masks(rgb_dir, sam2_predictor, points=args.points, negative_points=args.negative_points)
    kernel = np.ones((2, 2), np.uint8) 
    for i in range(len(masks)):
        mask_path = os.path.join(output_dir, "masked_rgb", f"mask_{i:04d}.png")
        mask = masks[i]
        # Normalize to 0~1
        if mask.max() > 1.0:
            mask = mask / 255.0
        # Further processing...
        # print(f"mask range: {mask.min()}, {mask.max()}")
        mask_path = os.path.join(output_dir, "masked_rgb", f"mask_{i:04d}.png")
        mask_uint8_img = (rgb_images[i]*mask[..., None]*255).astype(np.uint8)
        imageio.imwrite(mask_path, mask_uint8_img)
        # Save corresponding depth as npy
        depth = depth_images[i] * mask
        depth_path = os.path.join(output_dir, "depth", f"depth_{i:04d}.npy")
        np.save(depth_path, depth)
        depth_img_path = os.path.join(output_dir, "masked_depth_img", f"depth_{i:04d}.png")
        depth_vis = depth.copy()
        valid = np.isfinite(depth_vis) & (depth_vis > 0)
        if np.any(valid):
            depth_vis = (depth_vis - depth_vis[valid].min()) / (depth_vis[valid].max() - depth_vis[valid].min())
            depth_vis = (depth_vis * 255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(depth_vis, dtype=np.uint8)
        imageio.imwrite(depth_img_path, depth_vis)
            
    
    generate_videos(output_dir, fps=24)

    
    return rgb_images, depth_images, poses, intrinsics

def generate_videos(output_dir, fps=30):
    """Generate RGB and depth videos"""
    # Generate RGB video
    rgb_dir = os.path.join(output_dir, "masked_rgb")
    rgb_video_path = os.path.join(output_dir, "rgb_video.mp4")
    generate_video_from_images(rgb_dir, rgb_video_path, fps)
    
def generate_video_from_images(image_dir, video_path, fps=30):
    """Generate video from PNG images"""
    # Get all PNG files
    png_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    png_files.sort()  # Sort by filename
    
    if not png_files:
        print(f"Warning: No PNG files found in {image_dir}")
        return
    
    # Create video writer
    with imageio.get_writer(video_path, fps=fps) as writer:
        for png_file in png_files:
            img_path = os.path.join(image_dir, png_file)
            img = imageio.imread(img_path)
            writer.append_data(img)
    
    print(f"RGB video saved to: {video_path}")


def segment_gaussians(splats, classifier_dict):
    """Segment Gaussian points into foreground and background"""
    with torch.no_grad():
        # Get object_dc features
        objects_dc = splats['objects_dc']  # [N, 16]
        # Create the same model structure as during training
        classifier = torch.nn.Linear(3, 2)
        classifier.load_state_dict(classifier_dict)
        classifier.cuda()
        # Use classifier for prediction
        objects_dc = objects_dc.view(-1, 3)
        objects_dc = classifier(objects_dc)
        objects_dc_cls = torch.softmax(objects_dc, dim=-1)  # [N, 2]
        classification = objects_dc_cls.argmax(dim=-1)  # [N]
        
        # Get indices for foreground and background
        foreground_mask = classification == 1  # Assume 1 represents foreground
        background_mask = classification == 0  # Assume 0 represents background
        
        # Separate foreground and background Gaussian points
        foreground_splats = {}
        background_splats = {}
        
        for k, v in splats.items():
            foreground_splats[k] = v[foreground_mask].cpu()
            background_splats[k] = v[background_mask].cpu()
            
        return foreground_splats, background_splats

def save_gaussians(splats, save_path):
    """Save Gaussian points"""
    # Create save directory
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert all tensors to CPU and save
    save_dict = {
        'splats': {k: v.cpu() for k, v in splats.items()},
    }
    torch.save(save_dict, save_path)

def main():
    # Set up paths
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=False, help='Path to config file')
    args = parser.parse_args()
    if args.config_path is not None:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
        args.save_dir = config["work_dir"] + "/stage1_reconstruction"
        args.points = config["segment_config"]["points"]
        if "negative_points" in config["segment_config"]:
            args.negative_points = config["segment_config"]["negative_points"]
        else:
            args.negative_points = None
    save_dir = args.save_dir
    cam_pose_folder = os.path.join(save_dir, "camera_pose")
    ckpt_folder = os.path.join(save_dir, "3d/ckpts")
    masks_folder = os.path.join(save_dir, "3d/masks")  # Add masks folder path
    scene_info_path = os.path.join(save_dir, "scene_info.json")  # Add scene info file path
    
    # Load model
    print("Loading model...")
    ckpt = torch.load(os.path.join(ckpt_folder, "ckpt_9999.pt"))
    splats = ckpt["splats"]
    classifier = ckpt["classifier"]
    
    # Perform segmentation
    print("Performing segmentation...")
    foreground_splats, background_splats = segment_gaussians(splats, classifier)
    
    # Save segmentation results
    print("Saving segmentation results...")
    save_gaussians(foreground_splats, f"{save_dir}/foreground/foreground.pt")
    save_gaussians(background_splats, f"{save_dir}/background/background.pt")
    
    # Print statistics
    print(f"Segmentation completed!")
    print(f"Foreground Gaussian point count: {len(foreground_splats['means'])}")
    print(f"Background Gaussian point count: {len(background_splats['means'])}")
    
    # Generate RGB-D images
    print("Generating RGB-D images...")
    output_dir = f"{save_dir}/rgbd"
    rgb_images, depth_images, poses, intrinsics = generate_circular_rgbd_from_foreground(
        foreground_splats=foreground_splats,
        cam_pose_folder=cam_pose_folder,
        masks_folder=masks_folder,  # Pass masks folder path
        output_dir=output_dir,
        scene_info_path=scene_info_path,
        args=args
    )
    print(f"RGB-D image generation completed! Generated {len(rgb_images)} images")

if __name__ == "__main__":
    main() 