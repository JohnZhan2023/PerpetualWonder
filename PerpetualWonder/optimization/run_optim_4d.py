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

# from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

import warnings
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from toolkit.Scene_3d import Scene_3d
from toolkit.render import render_gs, render_concat_gs, render_2d_gs
from utils.loss import ssim
from scene.gaussian_model_4d import GaussianModel
warnings.filterwarnings("ignore")

background_color = [0, 0, 0]

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

    if not args.semi_round:
        front_view_id = len(args.camera_list) // 2
        camera_indices = [front_view_id]
    else:
        front_view_id = len(args.camera_list) // 2
        camera_indices = list(range(len(args.camera_list)))

    print(f"camera_indices: {camera_indices}")

    optimized_cam_list = [args.camera_list[i] for i in camera_indices]


    ###### ------------------ Path loading ------------------ ######
    video_path = args.video_path
    work_dir = args.work_dir

    scene_name = args.config["scene_name"]
    gaussians = GaussianModel(sh_degree=0, sub_gaussian = args.sub_gaussian, init_scale = args.init_scale, dt=args.dt, config=args.config)

    T = 49  
    frames_all = []
    for i, camera_id in enumerate(args.camera_list):
        if i not in camera_indices:
            continue
        video_dir = os.path.join(video_path, f"render_video_{camera_id}_flow/sdedit_{args.sdedit_strength[i]:.3f}/without_mask/output")
        frames = sorted(os.listdir(video_dir))
        frames = [os.path.join(video_dir, frame) for frame in frames]
        frames_this_traj = []
        for frame in frames:
            img = Image.open(frame).convert('RGB')
            new_img = Image.new('RGB', (1280, 704), (0, 0, 0))
            new_img.paste(img, (0, 0))
            resized_img = np.array(new_img)
            
            frame_image = torch.Tensor(resized_img).permute(2, 0, 1)
            frame_image = frame_image / 255.0
            frames_this_traj.append(frame_image)
        frames_all.append(frames_this_traj)

    optimize_iterations = args.optimize_iterations

    output_dir = args.output_dir
    gaussians_dir = os.path.join(output_dir, "gaussians") 
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gaussians_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True) 

    ###### ------------------ Load Scene ------------------ ######
    scene_3d = Scene_3d(work_dir, config=args.config)

    scene_3d.load_camera_params(args.camera_list)

    gaussians.load_normal_background(work_dir, T, add_foreground=True)


    if args.round_num > 1:
        gaussians.load_from_work_dir_further(work_dir, args.round_num)
    else:
        gaussians.load_from_work_dir(work_dir)

    output_dir = args.work_dir

    render_dir = os.path.join(output_dir, "stage3_optimization", "optimization_render", f"round_{args.round_num}")
    os.makedirs(render_dir, exist_ok=True)
    print("output dir:", render_dir)

    gaussians.training_4d_setup(T)
    pbar = tqdm(range(optimize_iterations))
    record_step = optimize_iterations // 2
    for iter_idx in pbar:
        if iter_idx % record_step == 0:
            for i, camera_id in enumerate(args.camera_list):
                img_dir = render_final_result(scene_3d, gaussians, render_dir, iter_idx, fix_camera=i)
            if args.semi_round:
                render_final_result(scene_3d, gaussians, render_dir, iter_idx, fix_camera=None)
        avg_visual_loss = 0
        avg_sim_loss = 0
        avg_total_loss = 0
        n_samples = 0
        for i, cam_id in enumerate(args.camera_list):
            if not args.semi_round and i != front_view_id:
                continue

            for tid in range(T):
                cam = scene_3d.get_camera_by_id(i)

                fg_means = gaussians.get_xyz(tid).float()
                fg_quats = gaussians.get_rotation(tid).float()
                fg_scales = gaussians.get_scaling(tid).float()
                fg_opacities = gaussians.get_opacity(tid).float()
                fg_colors = gaussians.get_features(tid).float()
                
                foreground_data = {
                    'means': fg_means,
                    'quats': fg_quats,
                    'scales': fg_scales,
                    'opacities': fg_opacities,
                    'colors': fg_colors
                }

                bg_means = gaussians.get_xyz_bg()
                bg_quats = gaussians.get_rotation_bg()
                bg_scales = gaussians.get_scaling_bg()
                bg_opacities = gaussians.get_opacity_bg(tid)
                bg_colors = gaussians.get_features_bg(tid)
                
                background_data = {
                    'means': bg_means,
                    'quats': bg_quats,
                    'scales': bg_scales,
                    'opacities': bg_opacities,
                    'colors': bg_colors
                }

                if gaussians.config["scene_name"] == "venice":
                    render_image, render_alpha = render_2d_gs(cam, foreground_data, background_data, background_color)  # [1, H, W, 3]
                else:
                    render_image, render_alpha = render_concat_gs(cam, foreground_data, background_data=background_data, background_color=background_color)  # [1, H, W, 3]
  
                render_image = render_image[..., :3]
                render_image = render_image.squeeze(0).permute(2, 0, 1)  # [3, H, W] for loss computation

                gt_image = frames_this_traj[tid].cuda()
                
                render_alpha = render_alpha.squeeze() # [H, W]

        

                if i == 1 or i == 2: 
                    l1_loss_val = F.l1_loss(render_image, gt_image) * args.front_l1_loss_weight
                    ssim_loss_val = (1 - ssim(render_image.unsqueeze(0), gt_image.unsqueeze(0))) * args.front_ssim_loss_weight  # SSIM loss
                    visual_loss = l1_loss_val + ssim_loss_val
                else:  
                    l1_loss_val = F.l1_loss(render_image, gt_image) * args.other_l1_loss_weight
                    ssim_loss_val = (1 - ssim(render_image.unsqueeze(0), gt_image.unsqueeze(0))) * args.other_ssim_loss_weight  # SSIM loss
                    visual_loss = l1_loss_val + ssim_loss_val
        
                xyzs = gaussians.get_xyz(tid)
                anchor_points = gaussians.get_anchor_points(tid)
                sim_loss = F.l1_loss(xyzs, anchor_points) * args.sim_loss_weight 

                loss = visual_loss + sim_loss

                avg_visual_loss += visual_loss.item()
                avg_sim_loss += sim_loss.item()
                avg_total_loss += loss.item()
                n_samples += 1
                
                loss.backward()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
        
        if n_samples > 0:
            avg_visual_loss /= n_samples
            avg_sim_loss /= n_samples
            avg_total_loss /= n_samples
            pbar.set_postfix({
                'L1': f'{avg_visual_loss:.4f}',
                'Sim': f'{avg_sim_loss:.4f}',
                'Total': f'{avg_total_loss:.4f}'
            })

    gaussians.save_generative_simulation(args.round_num)


def render_final_result(scene_3d, gaussians, output_dir, iter_idx, fix_camera=None):
    img_dir = os.path.join(output_dir, f"render_video_{iter_idx}_{fix_camera}")
    os.makedirs(img_dir, exist_ok=True)
    T = 49

    for tid in range(T):
        if fix_camera is not None:
            cam = scene_3d.get_camera_by_id(fix_camera) 
        else:
            cams = scene_3d.interpolate_camera_params(inter_num=25)
            redundancy = (len(cams) - T)//2
            cam = cams[tid+redundancy]

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
        
        bg_means = gaussians.get_xyz_bg(tid)
        bg_quats = gaussians.get_rotation_bg(tid)
        bg_scales = gaussians.get_scaling_bg(tid)
        bg_opacities = gaussians.get_opacity_bg(tid)  
        bg_colors = gaussians.get_features_bg(tid)
        
        background_data = {
            'means': bg_means,
            'quats': bg_quats,
            'scales': bg_scales,
            'opacities': bg_opacities,
            'colors': bg_colors
        }
        
        render_image, render_alpha = render_concat_gs(cam, foreground_data, background_data, background_color)  # [1, H, W, 3]
        

    images = []
    for tid in range(T):
        img_path = os.path.join(img_dir, f"{tid:04d}.png")
        images.append(imageio.imread(img_path))
        # rm the image
    
    imageio.mimsave(os.path.join(output_dir, f"render_video_{iter_idx:04d}_{fix_camera}.mp4"), images, fps=24)
    return img_dir


def render_complete_traj(scene_3d, gaussians, output_dir, round_num = 1):
    img_dir = os.path.join(output_dir, f"render_video_{round_num}")
    os.makedirs(img_dir, exist_ok=True)
    T = 49

    for tid in range(T):
        cams = scene_3d.interpolate_camera_params(inter_num=74)
        # redundancy = (len(cams) - T)//2
        tid_offset = tid + (round_num - 1) * T
        cam = cams[tid_offset]
            
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

        bg_means = gaussians.get_xyz_bg(tid)
        bg_quats = gaussians.get_rotation_bg(tid)
        bg_scales = gaussians.get_scaling_bg(tid)
        bg_opacities = gaussians.get_opacity_bg(tid) 
        bg_colors = gaussians.get_features_bg(tid)
        
        background_data = {
            'means': bg_means,
            'quats': bg_quats,
            'scales': bg_scales,
            'opacities': bg_opacities,
            'colors': bg_colors
        }
        

        render_image, render_alpha = render_concat_gs(cam, foreground_data, background_data, background_color)  # [1, H, W, 3]
    
        
        render_image = render_image[..., :3]

        render_image = render_image.squeeze(0)  # [H, W, 3]
    
        render_image = render_image.detach().cpu().numpy()  
        render_image = (render_image * 255).clip(0, 255).astype(np.uint8)  
        
        if render_image.shape[0] == 3:  
            render_image = render_image.transpose(1, 2, 0)
        
        imageio.imsave(os.path.join(img_dir, f"{tid:04d}.png"), render_image)
    
    images = []
    for tid in range(T):
        img_path = os.path.join(img_dir, f"{tid:04d}.png")
        images.append(imageio.imread(img_path))
        # rm the image
    
    imageio.mimsave(os.path.join(output_dir, f"render_video_{round_num}.mp4"), images, fps=24)
    return img_dir

def start_server():
    socketio.run(app, host="0.0.0.0", port=7776)


@socketio.on("connect")
def handle_connect():
    print("Client connected:", request.sid)
    global client_id
    client_id = request.sid


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected:", request.sid)
    global client_id
    client_id = None


@socketio.on("start")
def handle_start(data):
    print("Client connected:", request.sid)
    print("Received start signal.")
    start_event.set()  # Signal the main program to proceed


@socketio.on("gen")
def handle_gen(data):
    print("Received gen signal. Camera matrix: ", data)
    global view_matrix, keep_rendering
    keep_rendering = False
    view_matrix = data


@socketio.on("render-pose")
def handle_render_pose(data):
    global view_matrix_wonder, keep_rendering
    view_matrix_wonder = data


@socketio.on("scene-prompt")
def handle_new_prompt(data):
    assert isinstance(data, str)
    print("Received new scene prompt: " + data)
    global scene_name, change_scene_name_by_user
    scene_name = data
    change_scene_name_by_user = True



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=False,
                      help='Path to config file')
    parser.add_argument("--round_num", type=int, default=1, help='Round number')
    parser.add_argument("--semi_round", type=bool, default=False, help='Semi round')
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        args.work_dir = config["work_dir"]
        args.video_path = config["optimization_config"]["video_with_flow"]
        args.video_path = os.path.join(args.video_path, f"round_{args.round_num}")
        args.output_dir = config["work_dir"] + "/stage3_optimization/"
        if not args.semi_round:
            args.optimize_iterations = config["optimization_config"]["num_inference_steps"]
        else:
            args.optimize_iterations = config["optimization_config"]["num_inference_steps_semi_round"]
        args.camera_list = config["optimization_config"]["camera_list"]
        args.sdedit_strength = config["optimization_config"]["round_config"][args.round_num-1]["sdedit_strength"]
        args.sub_gaussian = config["sub-gaussian"]
        args.init_scale = config["simulator_config"]["particle_size"]
        args.sam_prompt = config["optimization_config"]["sam_prompt"]
        args.num_inference_steps_bg = config["optimization_config"]["num_inference_steps_bg"]
        args.front_l1_loss_weight = config["optimization_config"]["front_l1_loss_weight"]
        args.front_ssim_loss_weight = config["optimization_config"]["front_ssim_loss_weight"]
        args.other_l1_loss_weight = config["optimization_config"]["other_l1_loss_weight"]
        args.other_ssim_loss_weight = config["optimization_config"]["other_ssim_loss_weight"]
        args.sim_loss_weight = config["optimization_config"]["sim_loss_weight"]
        args.dt = float(config["simulator_config"]["dt"]) * float(config["simulator_config"]["substeps"])
        args.config = config
        print("args.camera_list", args.camera_list)
        print("args.sdedit_strength", args.sdedit_strength)

    # Start the server on a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True 
    server_thread.start()

    try:
        run(args)
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Error in main thread: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("Shutting down server...")
        os._exit(0) 
