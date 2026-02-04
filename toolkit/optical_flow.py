import torch
import math
from depth_diff_gaussian_rasterization_min import GaussianRasterizationSettings, GaussianRasterizer
import torch.nn.functional as F
import sys
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from torchvision.transforms import ToPILImage
import os

def proj_uv(xyz, cam):
    """
    Project 3D points onto the camera plane
    
    Args:
        xyz: torch.Tensor [N, 3] 3D points in world coordinates
        cam: Camera object containing camera intrinsics and extrinsics
        
    Returns:
        uv: torch.Tensor [N, 2] Projected 2D coordinates
    """
    device = xyz.device
    
    # 1. Transform world coordinates to camera coordinates
    # Convert 3D points to homogeneous coordinates
    N = xyz.shape[0]
    xyz_homo = torch.cat([xyz, torch.ones(N, 1, device=device)], dim=1)  # [N, 4]
    
    # Apply world-to-camera transformation
    worldtocam = cam.worldtocam.to(device)
    xyz_cam = (worldtocam @ xyz_homo.T).T  # [N, 4]
    xyz_cam = xyz_cam[:, :3]  # [N, 3]
    
    # 2. Project camera coordinates to image plane
    # Perspective division
    x_cam = xyz_cam[:, 0] / (xyz_cam[:, 2].clip(1e-6))  # Prevent division by zero
    y_cam = xyz_cam[:, 1] / (xyz_cam[:, 2].clip(1e-6))
    
    # Apply camera intrinsic matrix K
    K = cam.K.to(device)
    u = K[0, 0] * x_cam + K[0, 2]  # fx * x + cx
    v = K[1, 1] * y_cam + K[1, 2]  # fy * y + cy
    
    uv = torch.stack([u, v], dim=1)  # [N, 2]
    return uv



def render_optical_flow_2(camera, gs, prev_gs = None):

    viewmatrix, full_proj_transform, fovx, fovy, campos= camera.get_graphics_camera()
    height, width = camera.height, camera.width
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    bg_color = torch.ones(3, device="cuda:0", dtype=torch.float32)

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_transform,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=True  # Enable debug mode
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 1. Process positions
    means3D = gs['means']
    
    # 2. Process opacities
    opacities = gs['opacities']

    
    # 3. Process scales
    scales = gs['scales']
    
    # 4. Process rotations
    rotations = gs['quats']
    
    # 5. Process spherical harmonics coefficients
    shs = gs['colors']
    
    # For sh_degree=0, we only need the first SH coefficient (DC term)
    # shs should be [N, 1, 3] or [N, 3] for sh_degree=0
    if len(shs.shape) == 3 and shs.shape[1] > 1:
        # Take only the first coefficient (DC term) for sh_degree=0
        shs = shs[:, 0:1, :]  # [N, 1, 3]
    elif len(shs.shape) == 2 and shs.shape[1] == 3:
        # Already in correct format [N, 3], reshape to [N, 1, 3]
        shs = shs.unsqueeze(1)  # [N, 1, 3]
    
    # Basic settings
    means2D = torch.zeros(means3D.shape[0], 2, device=means3D.device, dtype=torch.float32, requires_grad=True)
    feats3D = torch.zeros(means3D.shape[0], 20, device=means3D.device, dtype=torch.float32)
    if prev_gs is None:
        delta_uv = torch.zeros_like(means2D)
    else:
        obj_xyz = gs["means"]
        prev_obj_xyz = prev_gs["means"]
        uv = proj_uv(obj_xyz, camera)
        prev_uv = proj_uv(prev_obj_xyz, camera)
        delta_uv = uv - prev_uv
        delta_uv = torch.cat([delta_uv, torch.zeros_like(delta_uv[:, :1], device=delta_uv.device)], dim=-1)
    rendered_image, _, feats, depth, flow = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        feats3D=feats3D,
        delta=delta_uv
    )
    # Process rendering results
    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
    

    return rendered_image, flow

def flow_to_image(flow):
    """
    Convert optical flow to visualization image
    
    Args:
        flow: torch.Tensor [H, W, 2] Optical flow field
        
    Returns:
        np.ndarray: [H, W, 3] Visualization image in BGR format
    """
    # Convert to numpy
    flow_np = flow.detach().cpu().numpy()
    
    # Calculate magnitude and angle of optical flow
    mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
    
    # Create HSV image
    hsv = np.zeros((flow_np.shape[0], flow_np.shape[1], 3), dtype=np.uint8)
    # Convert angle to hue (0-179)
    hsv[..., 0] = ang * 180 / np.pi / 2
    # Normalize magnitude and convert to saturation
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Set value to 255
    hsv[..., 2] = 255
    
    # Convert HSV to BGR
    flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_img



def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow field to RGB visualization image
    
    Args:
        flow: Optical flow field with shape (H, W, 2)
        
    Returns:
        RGB image with shape (H, W, 3)
    """
    # Calculate magnitude and direction of optical flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    
    # Map direction to hue H
    hsv[..., 0] = angle * 180 / np.pi / 2
    # Map magnitude to saturation S, normalized to [0, 255]
    hsv[..., 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Set value V to 255
    hsv[..., 2] = 255
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def compute_flow(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute optical flow between two frames
    
    Args:
        img1: First frame image
        img2: Second frame image
        
    Returns:
        Optical flow field
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        pyr_scale=0.5,    # Pyramid scale factor
        levels=3,         # Number of pyramid levels
        winsize=15,       # Window size
        iterations=3,     # Number of iterations
        poly_n=5,         # Polynomial expansion order
        poly_sigma=1.2,   # Gaussian standard deviation
        flags=0
    )
    
    return flow

def process_image_sequence(input_dir: str, output_dir: str):
    """
    Process image sequence, compute and visualize optical flow
    
    Args:
        input_dir: Input image directory
        output_dir: Output directory
    """
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob(os.path.join(input_dir, ext)))
    image_files = sorted(image_files)
    
    if len(image_files) < 2:
        print(f"Error: Found fewer than 2 images in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    

    for i in tqdm(range(len(image_files) - 1), desc="Processing image pairs"):
        img1 = cv2.imread(image_files[i])
        img2 = cv2.imread(image_files[i + 1])
        
        if img1 is None or img2 is None:
            print(f"Warning: Unable to read image {image_files[i]} or {image_files[i + 1]}")
            continue
            
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        flow = compute_flow(img1, img2)
    
        flow_path = os.path.join(output_dir, f'flow_{i}.npy')
        np.save(flow_path, flow)
