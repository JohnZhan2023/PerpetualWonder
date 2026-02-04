import gc
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
import imageio
import os
from datetime import datetime
import threading
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import matplotlib.pyplot as plt
from dataclasses import field
from typing import List

# from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import numpy as np
import torch
from tqdm import tqdm

import time
import cv2
import warnings

import sys
sys.path.append("./")
from scene.gaussian_model_4d import GaussianModel
from toolkit.Scene_3d import Scene_3d
from toolkit.render import render_concat_gs, render_2d_gs
from toolkit.optical_flow import render_optical_flow_2, flow_to_image

warnings.filterwarnings("ignore")


app = Flask(__name__)
CORS(app)  # Enable CORS on the Flask app
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for SocketIO

xyz_scale = 1000
client_id = None
scene_name = None
view_matrix = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_wonder = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device="cuda")
latest_frame = None
latest_viz = None
keep_rendering = True
iter_number = None
kf_gen = None
gaussians = None
opt = None
scene_dict = None
style_prompt = None
pt_gen = None
change_scene_name_by_user = False
exist_obj_names = None

sim = None
movement = None
already_object_pts_num = 0

# Event object used to control the synchronization
start_event = threading.Event()
gen_event = threading.Event()


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"running with seed: {seed}.")

def run(args):
    global client_id, view_matrix, scene_name, latest_frame, keep_rendering, kf_gen, latest_viz, gaussians, opt, background, scene_dict, style_prompt, pt_gen, change_scene_name_by_user
    global sim, movement, exist_obj_names

    ###### ------------------ Path loading ------------------ ######
    work_dir = args.work_dir

    cfg = args.config
    scene_name = cfg["scene_name"]
    sky_pic = f"examples/imgs/{scene_name}/sky.png"
    if os.path.exists(sky_pic):
        sky_pic = imageio.imread(sky_pic)
        sky_pic = np.array(sky_pic)
        sky_pic = sky_pic/255.0
        sky_pic = torch.from_numpy(sky_pic).to("cuda")
    else:
        sky_pic = None

    
    gaussians = GaussianModel(sh_degree=0, init_scale=args.init_scale, sub_gaussian=args.sub_gaussian, simulation_id=args.simulation_id, config=args.config)
    if args.round_num > 1:
        print(f"=========== load gs from {args.round_num - 1} round ==========")
        gaussians.load_normal_background(work_dir, add_foreground=True)
        gaussians.load_from_work_dir_further(work_dir, args.round_num)

    else:
        print("=========== initial gs ==========")
        gaussians.load_normal_background(work_dir, add_foreground=True)
        gaussians.load_from_work_dir(work_dir)
    # Set output directory
    output_dir = args.work_dir

    render_dir = os.path.join(output_dir, "stage2_forwardpass/render_output", f"round_{args.round_num}")
    os.makedirs(output_dir, exist_ok=True)

    T = 49  # Total number of frames


    ###### ------------------ Load Scene ------------------ ######
    scene_3d = Scene_3d(work_dir, config=args.config)

    scene_3d.load_camera_params(args.camera_list)

    for i, camera_id in enumerate(args.camera_list):
        flow_dir = os.path.join(render_dir, f"render_video_{args.camera_list[i]}_flow")
        os.makedirs(flow_dir, exist_ok=True)

        img_dir = render_final_result(scene_3d, gaussians, render_dir, flow_dir, fix_camera=i, sky_pic=sky_pic)

    # render_final_result(scene_3d, gaussians, render_dir,flow_dir=None, fix_camera=None)



def render_final_result(scene_3d, gaussians, output_dir, flow_dir, fix_camera=None, sky_pic=None):
    T = 49
    # camera_list = [0, 10, 65, 108, 45]
    camera_list = scene_3d.camera_list
    # scene_3d.load_camera_params(camera_list)
    if fix_camera is not None:
        img_dir = os.path.join(output_dir, f"render_video_{camera_list[fix_camera].img_id}")
    else:
        img_dir = os.path.join(output_dir, f"render_video_all")

    os.makedirs(img_dir, exist_ok=True)
    cams = scene_3d.interpolate_camera_params(inter_num=13)
    T = 49


    for tid in range(0, T):
        if fix_camera is not None:
            cam = scene_3d.get_camera_by_id(fix_camera)  # Use camera for corresponding frame
        else:
            cam = cams[tid]
        # Get foreground Gaussian data and print statistics
        fg_means = gaussians.get_xyz(tid)
        fg_quats = gaussians.get_rotation(tid)
        fg_scales = gaussians.get_scaling(tid)
        fg_opacities = gaussians.get_opacity(tid)
        fg_colors = gaussians.get_features(tid)
        
        foreground_data = {
            'means': fg_means,
            'quats': fg_quats,
            'scales': fg_scales,
            'opacities': fg_opacities,
            'colors': fg_colors
        }
        # print(f"foreground_data: {foreground_data['means'].shape}, {foreground_data['quats'].shape}, {foreground_data['scales'].shape}, {foreground_data['opacities'].shape}, {foreground_data['colors'].shape}")
        if tid > 0:
            prev_means = gaussians.get_xyz(tid - 1)
            prev_quats = gaussians.get_rotation(tid - 1)
            prev_scales = gaussians.get_scaling(tid - 1)
            prev_opacities = gaussians.get_opacity(tid - 1)
            prev_colors = gaussians.get_features(tid - 1)
            if prev_means.shape[0] != foreground_data['means'].shape[0]:
                source = [0, 0, 0]
                # Calculate number of particles to add
                current_count = foreground_data['means'].shape[0]
                prev_count = prev_means.shape[0]
                diff_count = current_count - prev_count
                
                if diff_count > 0:
                    # Current frame has more particles, need to extend previous frame data
                    device = prev_means.device
                    
                    # Fill means with source position
                    source_tensor = torch.tensor(source, device=device, dtype=prev_means.dtype).repeat(diff_count, 1)
                    prev_means = torch.cat([prev_means, source_tensor], dim=0)
                    
                    # Fill other attributes with 0 or default values
                    # Quaternion: unit quaternion [1, 0, 0, 0]
                    zero_quats = torch.zeros(diff_count, 4, device=device, dtype=prev_quats.dtype)
                    zero_quats[:, 0] = 1.0  # w component is 1
                    prev_quats = torch.cat([prev_quats, zero_quats], dim=0)
                    
                    # Scale: very small value
                    zero_scales = torch.ones(diff_count, 3, device=device, dtype=prev_scales.dtype) * 0.001
                    prev_scales = torch.cat([prev_scales, zero_scales], dim=0)
                    
                    # Opacity: 0
                    zero_opacities = torch.zeros(diff_count, device=device, dtype=prev_opacities.dtype)
                    prev_opacities = torch.cat([prev_opacities, zero_opacities], dim=0)
                    
                    # Color: black or transparent
                    zero_colors = torch.zeros(diff_count, prev_colors.shape[1], prev_colors.shape[2], device=device, dtype=prev_colors.dtype)
                    prev_colors = torch.cat([prev_colors, zero_colors], dim=0)
                
                
                elif diff_count < 0:
                    # Current frame has fewer particles, truncate previous frame data
                    prev_means = prev_means[:current_count]
                    prev_quats = prev_quats[:current_count]
                    prev_scales = prev_scales[:current_count]
                    prev_opacities = prev_opacities[:current_count]
                    prev_colors = prev_colors[:current_count]
                    
                    print(f"Truncated {-diff_count} particles")

            prev_gs = {
                "means": prev_means,
                "quats": prev_quats,
                "scales": prev_scales,
                "opacities": prev_opacities,
                "colors": prev_colors
            }
            
            # Validate data shapes before rendering
            if foreground_data['means'].shape[0] != prev_gs['means'].shape[0]:
                raise ValueError(f"Particle count mismatch: foreground={foreground_data['means'].shape[0]}, prev={prev_gs['means'].shape[0]}")
            if foreground_data['means'].shape[1] != 3 or prev_gs['means'].shape[1] != 3:
                raise ValueError(f"Invalid means shape: foreground={foreground_data['means'].shape}, prev={prev_gs['means'].shape}")
            
            render_image_flow, flow = render_optical_flow_2(cam, foreground_data, prev_gs)
            # Save optical flow data
            if flow is not None and flow_dir is not None:
                flow_np = flow.detach().cpu().numpy()
                flow_path = os.path.join(flow_dir, f"flow_{tid}.npy")
                np.save(flow_path, flow_np)
                flow_vis = flow.permute(1, 2, 0)
                flow_img = flow_to_image(flow_vis)
                cv2.imwrite(os.path.join(flow_dir, f"flow_{tid}.png"), flow_img)    
        
        # Get background Gaussian data
        bg_means = gaussians.get_xyz_bg()
        bg_quats = gaussians.get_rotation_bg()
        bg_scales = gaussians.get_scaling_bg()
        bg_opacities = gaussians.get_opacity_bg()
        bg_colors = gaussians.get_features_bg()

        
        background_data = {
            'means': bg_means,
            'quats': bg_quats,
            'scales': bg_scales,
            'opacities': bg_opacities,
            'colors': bg_colors
        }

        
        # Black background
        background_color = [0, 0, 0]

        # Use mixed rendering: foreground 3DGS + background 2DGS
        if gaussians.config["scene_name"] == "venice":
            render_image, render_alpha = render_2d_gs(cam, foreground_data, background_data, background_color)  # [1, H, W, 3]
        else:
            render_image, render_alpha = render_concat_gs(cam, foreground_data, background_data=background_data, background_color=background_color)  # [1, H, W, 3]
        render_image = render_image[..., :3]
        # merge with the sky pic
        # render_image = render_alpha * render_image + (1 - render_alpha) * sky_pic
        render_image = render_image.squeeze(0)  # [H, W, 3]
        render_image = render_image.detach().cpu().numpy()  # Convert to numpy array
        # analyze the render_image's value range
        # print(f"render_image's value range: {render_image.min()}, {render_image.max()}")
        render_image = (render_image * 255).clip(0, 255).astype(np.uint8)  # Convert to 0-255 range
            
        
        # Ensure image dimensions are [H, W, 3]
        if render_image.shape[0] == 3:  # If dimensions are [3, H, W]
            render_image = render_image.transpose(1, 2, 0)
        
        # Save as PNG image
        imageio.imsave(os.path.join(img_dir, f"{tid:04d}.png"), render_image)
    
    # If needed, can combine all PNGs into a video
    images = []
    for tid in range(T):
        img_path = os.path.join(img_dir, f"{tid:04d}.png")
        images.append(imageio.imread(img_path))
        # Remove the image
    
    # Save video
    if fix_camera is not None:
        imageio.mimsave(os.path.join(output_dir, f"render_video_{camera_list[fix_camera].img_id}.mp4"), images, fps=24)
    else:
        imageio.mimsave(os.path.join(output_dir, f"render_video_all.mp4"), images, fps=24)
    return img_dir


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--work_dir", type=str, default="3d_result/jam_4")
    # parser.add_argument("--video_path", type=str, default="3d_result/jam_4_flow/sdedit_0.850/without_mask")
    # parser.add_argument("--output_dir", type=str, default="3d_result/jam_4_optimize_3d")
    # parser.add_argument("--optimize_iterations", type=int, default=501)
    parser.add_argument("--config", type=str, required=False,
                      help='Path to config file')
    parser.add_argument("--round_num", type=int, default=1, help='Round number')
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        args.work_dir = config["work_dir"]
        # args.camera_list = config["simulator_config"]["camera_list"]
        args.camera_list = [83, 121, 159]
        args.sub_gaussian = config["sub-gaussian"]
        args.init_scale = config["simulator_config"]["particle_size"]
        args.simulation_id = config["simulator_config"]["simulation_entity_id"]
        args.config = config
        print("args.camera_list", args.camera_list)


    try:
        # Run main optimization loop
        run(args)
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Error in main thread: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("Shutting down server...")
        # Cleanup code can be added here
        os._exit(0)  # Force exit all threads
