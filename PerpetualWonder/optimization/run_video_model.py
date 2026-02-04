
"""
Video refinement using SDEdit method for Gaussian scene optimization.

This module uses SDEdit (Score-based Diffusion Editing) to generate improved videos 
that are used to refine the Gaussian scene. The refinement process enhances the 
quality of the generated videos by applying diffusion-based editing techniques.

Output:
    Running this script generates refined videos under:
    3d_result/{scene_name}/stage3_optimization/go_with_flow/
    
    The output includes videos processed with different SDEdit strengths and 
    optical flow information for further Gaussian scene optimization.
"""

import rp
import shutil
import os


rp.r._pip_import_autoyes=True #Automatically install missing packages

rp.pip_import('fire')
rp.git_import('CommonSource') #If missing, installs code from https://github.com/RyannDaGreat/CommonSource
import sys
sys.path.append("./")
import video_models.noise_warp as nw
from video_models.cut_and_drag_inference_with_masked_sdedit import get_pipe, load_sample_cartridge, run_pipe
import fire
import argparse
import numpy as np

import cv2
import numpy as np
from PIL import Image

def resize_flow(flow, new_size=(480, 720)):
    """
    Resize a flow field of shape (2, H, W) to (2, new_H, new_W).

    Parameters:
        flow (numpy.ndarray): Flow data of shape (2, H, W).
        new_size (tuple): Desired output size (new_H, new_W).

    Returns:
        numpy.ndarray: Resized flow of shape (2, new_H, new_W).
    """
    resized_flow = np.zeros((2, new_size[1], new_size[0]), dtype=flow.dtype)

    for i in range(2):  # Resize each flow channel separately
        resized_flow[i] = cv2.resize(flow[i], new_size, interpolation=cv2.INTER_LINEAR)

    return resized_flow

def main_warp_noise(args):
    """
    Takes a video URL or filepath and an output folder path
    It then resizes that video to height=480, width=720, 49 frames (CogVidX's dimensions)
    Then it calculates warped noise at latent resolution (i.e. 1/8 of the width and height) with 16 channels
    It saves that warped noise, optical flows, and related preview videos and images to the output folder
    The main file you need is <output_folder>/noises.npy which is the gaussian noises in (H,W,C) form
    """

    # should be output_video/simulation_name/traj_name
    output_folder = args.output_folder
    # should be simulation_name/simulation/
    input_folder = args.input_folder
    
    os.makedirs(output_folder, exist_ok=True)
    
    crop_start = args.crop_start
    FLOW = 2 ** 3
    LATENT = 8

    input_flow = False
    video = args.input_flow_path
    if rp.is_a_folder(video):
        FRAME = 1
        FLOW = 2 ** 2
        frame_flows = sorted([os.path.join(video, f) for f in os.listdir(video) if f.endswith('.npy')])
        frame_flows = [np.load(flow_file) for flow_file in frame_flows]
        for i in range(len(frame_flows)):
            frame_flows[i][..., 0] = frame_flows[i][..., 0] * 2 * 1280 - 1280
            frame_flows[i][..., 1] = frame_flows[i][..., 1] * 2 * 704 - 704
        
        print("Number of frames for optical flow:", len(frame_flows))
        print("Shape of each frame:", frame_flows[0].shape) # ( 704, 1280, 2)

        frame_flows = frame_flows[0:48] # rp.resize_list(frame_flows, length=48)
        for i in range(len(frame_flows)):
            frame_flows[i] = resize_flow(frame_flows[i]) # （720, 720, 2）

        for i in range(len(frame_flows)):
            frame_flows[i][0] = frame_flows[i][0] * (720 / 1280) # （720, 720, 2）
            frame_flows[i][1] = frame_flows[i][1] * (480 / 704) # （720, 720, 2）


        frame_flows = rp.as_numpy_array(frame_flows)
        frame_flows = frame_flows.transpose(0, 1, 3, 2)
        print("Shape of optical flow:", frame_flows.shape)
        video = frame_flows
        input_flow = True
        
    else:
        raise ValueError(f"The given video={repr(video)} is not a optical flow folder or a video file")

    #See this function's docstring for more information!
    output = nw.get_noise_from_video(
        video,
        remove_background=False, #Set this to True to matte the foreground - and force the background to have no flow
        visualize=True,          #Generates nice visualization videos and previews in Jupyter notebook
        save_files=True,         #Set this to False if you just want the noises without saving to a numpy file
        input_flow=input_flow,
        noise_channels=16,
        output_folder=output_folder,
        resize_frames=FRAME,
        resize_flow=FLOW,
        downscale_factor= round(FRAME * FLOW) * LATENT,
        device="cuda"
    )

    print("Noise shape:"  ,output.numpy_noises.shape)
    print("Output folder:",output.output_folder)

    # Link to render_video.mp4
    video_src = args.input_video_path
    video_name = os.path.basename(video_src)
    video_dst = os.path.join(output_folder, "input.mp4")
    shutil.copy(video_src, video_dst)

    return


def main_cut_and_drag(args):
    output_folder = args.output_folder
    input_folder = args.input_folder

    crop_start = args.crop_start # cartridge
    num_inference_steps = args.num_inference_steps # cartridge
    degradation = args.degradation # cartridge

    prompt = args.prompt # cartridge

    model_name = "I2V5B_final_i38800_nearest_lora_weights"
    low_vram = True
    device = "cuda"
    pipe = get_pipe(model_name=model_name, device=device, low_vram=low_vram)

    sdedit_strengths = args.sdedit_strengths
    # 完全禁用mask功能
    mask_strength_gaps = [-1]  # 只使用-1，表示不使用mask
    print("Running with sdedit_strength:", sdedit_strengths)
    print("Running without mask (mask_strength_gap: -1)")
    
    for sdedit_strength in sdedit_strengths:
        # 只运行不使用mask的情况
        mask_sdedit_strength = -1
        exp_output_folder = os.path.join(output_folder, f"sdedit_{sdedit_strength:.03f}", f"without_mask")
        print(f"Running with sdedit_strength={sdedit_strength} and no mask")
        os.makedirs(exp_output_folder, exist_ok=True)
        output_mp4_path = os.path.join(exp_output_folder, f"output.mp4")

        sample_path = output_folder
        noise_downtemp_interp = "nearest"
        # 使用None让函数自动选择第一帧作为图像
        # get the camera_id from the input_folder
        
        camera_id = int(args.input_flow_path.split("/")[-1].split("_")[-2])
        if camera_id == 121:
            image = f"examples/imgs/{args.scene_name}/{args.scene_name}.png"
        else:
            image = os.path.join(args.input_folder, f"stage1_reconstruction/3d/images/{camera_id:06d}.jpg")
        if args.round_num > 1:
            id = args.whole_camera_list.index(camera_id)
            image = os.path.join(args.input_folder, f"stage3_optimization/go_with_flow/round_{args.round_num-1}/render_video_{camera_id}_flow/sdedit_{args.pre_sdedit_strengths[id]:.3f}/without_mask/output/0048.png")


        guidance_scale = 6
        mask_strength = -1  # 确保mask_strength为-1

        cartridge_kwargs = rp.broadcast_kwargs(
            rp.gather_vars(
                "sample_path", # done
                "degradation", # done
                "noise_downtemp_interp", # done
                "image", # done
                "prompt", # done
                "num_inference_steps", # done
                "guidance_scale", # done
                "sdedit_strength", # done
                "mask_strength", # done
                "crop_start", # done
                # "v2v_strength",
            )
        )

        cartridges = rp.load_files(lambda x:load_sample_cartridge(**x, model_name=model_name), cartridge_kwargs, show_progress='eta:Loading Cartridges')
        for cartridge in cartridges:
            pipe_out = run_pipe(
                pipe=pipe,
                cartridge=cartridge,
                output_mp4_path=output_mp4_path,
            )

def main():
    parser = argparse.ArgumentParser(description='Process video and generate warped noise.')
    parser.add_argument('--config', type=str, required=False,
                      help='Path to config file')
    # parser.add_argument('--video', type=str, required=True,
    #                   help='Video URL or filepath')
    parser.add_argument('--input_folder', type=str, required=False,
                      help='Path to input folder, for the entire simulation output')
    parser.add_argument('--output_folder', type=str, required=False,
                      help='Path to output folder')
    parser.add_argument('--crop_start', type=int, default=120,
                      help='Where to start cropping (default: 120)')
    parser.add_argument('--num_inference_steps', type=int, default=25,
                      help='Number of inference steps (default: 25)')
    parser.add_argument('--degradation', type=float, default=0.4,
                      help='Degradation level (default: 0.4)')
    parser.add_argument('--sdedit_strengths', type=float, nargs='+', default=[0.75, 0.8, 0.85],
                      help='SDEdit strength value(s). Can be a single float or list of floats.')
    parser.add_argument('--round_num', type=int, default=1, help='Round number')
    args = parser.parse_args()
    input_flow_list = []
    input_video_list = []
    if args.config is not None:
        with open(args.config, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        args.input_folder = config["work_dir"]
        output_folder = config["video_config"]["output_folder"]
        args.sdedit_strengths = config["video_config"]["sdedit_strengths"]
        if args.round_num > 1:
            args.pre_sdedit_strengths = config["optimization_config"]["round_config"][args.round_num-2]["sdedit_strength"]
        else:
            args.pre_sdedit_strengths = args.sdedit_strengths
        camera_list = config["simulator_config"]["camera_list"]
        args.camera_list = camera_list
        args.whole_camera_list = config["optimization_config"]["camera_list"]
        args.prompt = config["optimization_config"]["round_config"][args.round_num-1]["prompt"]
        print("prompt: ", args.prompt)
        args.scene_name = config["scene_name"]
        for camera_id in camera_list:
            input_flow_list.append(os.path.join(args.input_folder, 'stage2_forwardpass', 'render_output', f'round_{args.round_num}', f'render_video_{camera_id}_flow'))
            if camera_id == 121:
                input_video_list.append(os.path.join(args.input_folder, 'stage2_forwardpass', 'render_output', f'round_{args.round_num}', f'render_video_{camera_id}.mp4'))
            elif camera_id == 83:
                input_video_list.append(os.path.join(args.input_folder, 'stage3_optimization', 'optimization_render', f'round_{args.round_num}', f'render_video_0200_0.mp4'))
            elif camera_id == 159:
                input_video_list.append(os.path.join(args.input_folder, 'stage3_optimization', 'optimization_render', f'round_{args.round_num}', f'render_video_0200_2.mp4'))
            output_folder = os.path.join(args.input_folder, 'stage3_optimization', 'go_with_flow', f'round_{args.round_num}')
    for input_flow, input_video in zip(input_flow_list, input_video_list):
        args.input_flow_path = input_flow
        args.input_video_path = input_video
        args.output_folder = os.path.join(output_folder, os.path.basename(input_flow))
        main_warp_noise(args)
        
        main_cut_and_drag(args)

if __name__ == "__main__":
    main()