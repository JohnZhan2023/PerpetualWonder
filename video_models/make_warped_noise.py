#Ryan Burgert 2024

#Setup:
#    Run this in a Jupyter Notebook on a computer with at least one GPU
#        `sudo apt install ffmpeg git`
#        `pip install rp`
#    The first time you run this it might be a bit slow (it will download necessary models)
#    The `rp` package will take care of installing the rest of the python packages for you

import rp
import shutil
import os

rp.r._pip_import_autoyes=True #Automatically install missing packages

rp.pip_import('fire')
rp.git_import('CommonSource') #If missing, installs code from https://github.com/RyannDaGreat/CommonSource
import noise_warp as nw
import fire
import numpy as np

import cv2
import numpy as np

def resize_flow(flow, new_size=(720, 720)):
    """
    Resize a flow field of shape (2, H, W) to (2, new_H, new_W).

    Parameters:
        flow (numpy.ndarray): Flow data of shape (2, H, W).
        new_size (tuple): Desired output size (new_H, new_W).

    Returns:
        numpy.ndarray: Resized flow of shape (2, new_H, new_W).
    """
    resized_flow = np.zeros((2, new_size[0], new_size[1]), dtype=flow.dtype)

    for i in range(2):  # Resize each flow channel separately
        resized_flow[i] = cv2.resize(flow[i], new_size, interpolation=cv2.INTER_LINEAR)

    return resized_flow

def main(video:str, output_folder:str, first_frame:str=None, sim_name:str=None, crop_start = 120):
    """
    Takes a video URL or filepath and an output folder path
    It then resizes that video to height=480, width=720, 49 frames (CogVidX's dimensions)
    Then it calculates warped noise at latent resolution (i.e. 1/8 of the width and height) with 16 channels
    It saves that warped noise, optical flows, and related preview videos and images to the output folder
    The main file you need is <output_folder>/noises.npy which is the gaussian noises in (H,W,C) form
    """

    if rp.folder_exists(output_folder):
        user_input = input(f"The given output_folder={repr(output_folder)} already exists! Do you want to clear this folder and continue? (yes/no): ").strip().lower()
        if user_input == 'yes':
            shutil.rmtree(output_folder)
        else:
            raise RuntimeError(f"Please specify a folder that doesn't exist or choose to clear the existing folder.")

    if first_frame is not None:
        first_frame = rp.load_image(first_frame)
        first_frame = rp.resize_image_to_hold(first_frame, height=480, width=720)
        first_frame = first_frame[crop_start:crop_start + 480, :, :]
        first_frame = rp.as_numpy_array(first_frame)
        rp.save_image(first_frame, rp.path_join("./resized_gt/", f'{sim_name}.png'))
        print("Resized and cropped first frame saved to", rp.path_join("./resized_gt/", f'{sim_name}.png'))


    FLOW = 2 ** 3
    LATENT = 8

    input_flow = False
    if rp.is_a_folder(video):
        FRAME = 1
        FLOW = 2 ** 2
        frame_flows = sorted([os.path.join(video, f) for f in os.listdir(video) if f.endswith('.npy')])
        frame_flows = [np.load(flow_file) for flow_file in frame_flows]
        for i in range(len(frame_flows)):
            frame_flows[i] = frame_flows[i] * 2 * 512 - 512
        
        print("Number of frames for optical flow:", len(frame_flows))
        print("Shape of each frame:", frame_flows[0].shape)

        frame_flows = frame_flows[1:49] # rp.resize_list(frame_flows, length=48)
        for i in range(len(frame_flows)):
            frame_flows[i] = resize_flow(frame_flows[i])
        for i in range(len(frame_flows)):
            frame_flows[i] = frame_flows[i] * (720 / 512)
        for i in range(len(frame_flows)):
            frame_flows[i] = frame_flows[i][:, crop_start:crop_start + 480, :]

        # masks_path = video.replace("flows_actual", "masks_obj_0000")
        # masks = sorted([os.path.join(masks_path, f) for f in os.listdir(masks_path) if f.endswith('.png')])[1:49]
        # masks = [rp.load_image(mask) for mask in masks]
        
        # resized_masks = np.zeros((len(masks), 720, 720), dtype=masks[0].dtype)
        # for i in range(len(masks)):
        #     resized_masks[i] = cv2.resize(masks[i], (720, 720), interpolation=cv2.INTER_NEAREST)
        # cropped_masks = resized_masks[:, crop_start:crop_start + 480, :] / 255

        # for i in range(len(cropped_masks)):
        #     frame_flows[i][:, cropped_masks[i] > 0.5] *= 1.0

        frame_flows = rp.as_numpy_array(frame_flows)
        print("Shape of optical flow:", frame_flows.shape)
        video = frame_flows
        input_flow = True
        
    elif rp.is_video_file(video):
        FRAME = 2**-1
        video=rp.load_video(video)
        #Preprocess the video
        video=rp.resize_list(video,length=49) #Stretch or squash video to 49 frames (CogVideoX's length)
        video=rp.resize_images_to_hold(video,height=480,width=720)
        for i in range(len(video)):
            video[i] = video[i][crop_start:crop_start + 480, :, :]
        video=rp.as_numpy_array(video)
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
    )

    print("Noise shape:"  ,output.numpy_noises.shape)
    print("Output folder:",output.output_folder)

if __name__ == "__main__":
    fire.Fire(main) 