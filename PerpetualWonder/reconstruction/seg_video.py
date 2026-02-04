#!/usr/bin/env python3
"""
Video segmentation script using SAM2 for foreground identification.

This module uses SAM2 (Segment Anything Model 2) to identify and segment foreground 
objects in videos. It generates masks for each frame that can be used for subsequent 
processing steps such as Gaussian splatting reconstruction.

Usage:
    python seg_video.py --config_path examples/configs/play_doh.yaml
    python seg_video.py --video_path video.mp4 --output_dir results/

Output:
    Running this script generates the following files and directories under 
    3d_result/{scene_name}/stage1_reconstruction/:
    - 3d/masks/: Directory containing binary mask images for each frame
    - segmentation_result.mp4: Video file with segmentation masks overlaid on original frames
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import sys
import argparse
import yaml
sys.path.append("./")
from toolkit.utils import images_to_mp4
sys.path.append("submodules/segment_anything")
from sam2.build_sam import build_sam2_video_predictor

def load_video(video_path):
    """Load video file"""
    # Check if video can be opened
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    cap.release()
    return video_path

def generate_masks(video_path, sam2_predictor, points=None, negative_points=None, args=None):
    """Generate masks using SAM2"""
    if not video_path.endswith(".mp4"):
        images_to_mp4(video_path, os.path.join(os.path.dirname(video_path), "multiview.mp4"), fps=24)
        video_path = os.path.join(os.path.dirname(video_path), "multiview.mp4")
        if args is not None:
            args.video_path = video_path
    
    # Initialize SAM2 (using video file mode)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam2_predictor.init_state(video_path)

        labels = np.ones(len(points), dtype=np.int32)  # All points are foreground points
        if negative_points is not None:
            labels = np.concatenate([labels, np.zeros(len(negative_points), dtype=np.int32)])
            points = np.concatenate([points, negative_points])

        # Add prompt
        frame_idx, object_ids, mask_logits = sam2_predictor.add_new_points_or_box(
            state, 
            0,  # First frame
            obj_id=1,
            points=points,
            labels=labels
        )


        
        # Propagate to entire video
        masks_dict = {}
        
        for frame_idx, object_ids, mask_logits in tqdm(
            sam2_predictor.propagate_in_video(state),
            desc="Propagating masks"
        ):
            if len(mask_logits) > 0:
                # Use lower threshold to include more foreground regions
                mask = (mask_logits[0] > -0.5).cpu().numpy()  # Lower threshold to include more foreground
                # Ensure mask is 2D
                if mask.ndim == 3:
                    mask = mask[0]  # Take first channel
                    
                mask = mask.astype(np.uint8) * 255
                masks_dict[frame_idx] = mask
                
        return masks_dict


def propagate_mask_from_first_frame(video_path, first_frame_mask_path, sam2_predictor, args=None):
    """
    Propagate an existing first frame mask directly to subsequent frames
    
    Args:
        video_path: Path to video file
        first_frame_mask_path: Path to first frame mask image
        sam2_predictor: SAM2 predictor
        args: Optional arguments
    
    Returns:
        masks_dict: Dictionary containing masks for all frames {frame_idx: mask}
    """
    if not video_path.endswith(".mp4"):
        from toolkit.utils import images_to_mp4
        images_to_mp4(video_path, os.path.join(os.path.dirname(video_path), "multiview.mp4"), fps=24)
        video_path = os.path.join(os.path.dirname(video_path), "multiview.mp4")
        if args is not None:
            args.video_path = video_path
    
    # Load first frame mask
    if not os.path.exists(first_frame_mask_path):
        print(f"Error: First frame mask file does not exist: {first_frame_mask_path}")
        return {}
    
    first_mask = cv2.imread(first_frame_mask_path, cv2.IMREAD_GRAYSCALE)
    if first_mask is None:
        print(f"Error: Cannot load first frame mask: {first_frame_mask_path}")
        return {}
    
    # Convert mask to binary mask (True/False)
    first_mask_binary = (first_mask > 0)  # Keep as boolean type
    
    # Ensure mask is 2D
    if first_mask_binary.ndim == 3:
        first_mask_binary = first_mask_binary[:, :, 0]  # Take first channel
    
    print(f"First frame mask shape: {first_mask_binary.shape}, dtype: {first_mask_binary.dtype}")
    
    # Initialize SAM2 (using video file mode)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam2_predictor.init_state(video_path)
        
        # Convert first frame mask to torch tensor (must be 2D bool type)
        first_mask_tensor = torch.from_numpy(first_mask_binary).bool()

        
        # Use add_new_mask method to add first frame mask
        frame_idx, object_ids, mask_logits = sam2_predictor.add_new_mask(
            state,
            frame_idx=0,  # First frame
            obj_id=1,
            mask=first_mask_tensor
        )
        
        print(f"First frame mask added, starting propagation to subsequent frames...")
        
        # Propagate to entire video
        masks_dict = {}
        
        for frame_idx, object_ids, mask_logits in tqdm(
            sam2_predictor.propagate_in_video(state),
            desc="Propagating from first frame mask"
        ):
            if len(mask_logits) > 0:
                # Use threshold to convert logits to binary mask
                mask = (mask_logits[0] > 0.0).cpu().numpy()
                # Ensure mask is 2D
                if mask.ndim == 3:
                    mask = mask[0]  # Take first channel
                    
                mask = mask.astype(np.uint8) * 255
                masks_dict[frame_idx] = mask
        
        print(f"Mask propagation completed, processed {len(masks_dict)} frames")
        return masks_dict

def save_results(video_path, masks_dict, output_dir, sky = False):
    """Save segmentation results, including video and binary masks"""
    print("Saving results...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    if not sky:
        mask_dir = os.path.join(output_dir, "stage1_reconstruction/3d/masks")
        # Define video save path
        output_path = os.path.join(output_dir, "stage1_reconstruction/segmentation_result.mp4")
    else:
        mask_dir = os.path.join(output_dir, "stage1_reconstruction/3d/sky_masks")
        output_path = os.path.join(output_dir, "sky_segmentation_result.mp4")
    os.makedirs(mask_dir, exist_ok=True)
    
    # Read video to get parameters
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 24 # Prevent case where fps cannot be obtained
    
    # --- Core fix: Initialize out object ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or use 'avc1'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # -------------------------------

    # Process each frame
    # Note: Using range(len(masks_dict)) might be problematic since masks_dict keys are frame_idx
    # Recommend iterating through video until end, or iterate based on max index in masks_dict
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_idx in tqdm(range(num_frames), desc="Saving video and masks"):
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in masks_dict:
            # Get mask and ensure it is 2D
            mask = masks_dict[frame_idx]
            if mask.ndim == 3:
                mask = mask[0]
            
            # Resize mask to match video frame
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Create binary mask
            foreground_mask = (mask > 0).astype(np.uint8) * 255
            
            # Save binary mask image
            cv2.imwrite(os.path.join(mask_dir, f"{frame_idx:03d}.png"), foreground_mask)
            
            # Create colored overlay for video visualization
            colored_mask = np.zeros_like(frame)
            if not sky:
                colored_mask[foreground_mask > 0] = [0, 255, 0]  # Green for foreground
            else:
                colored_mask[foreground_mask > 0] = [255, 0, 0]  # Blue for sky
            
            # Overlay mask on original frame
            alpha = 0.5
            frame = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)
        
        # Write video frame
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video saved to: {output_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=False, help="Input config file path")
    parser.add_argument("--video_path", type=str, required=False, help="Input video path")
    parser.add_argument("--output_dir", type=str, required=False, help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="./submodules/sam2/checkpoints/sam2.1_hiera_large.pt", help="SAM2 model checkpoint path")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 model config file path")
    args = parser.parse_args()
    if args.config_path is not None:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
        args.video_path = config["segment_config"]["video_path"]
        args.output_dir = config["work_dir"]
        args.points = config["segment_config"]["points"]
        args.scene_name = config["scene_name"]
        if "negative_points" in config["segment_config"]:
            negative_points = config["segment_config"]["negative_points"]
            print(f"negative_points: {negative_points}")
        else:
            negative_points = None
    
    # Initialize SAM2
    sam2_predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint)
    
    # Generate foreground masks
    print("Segmenting foreground...")
    foreground_masks_dict = generate_masks(args.video_path, sam2_predictor, args=args, points=args.points, negative_points=negative_points)
    print(args.video_path)
    save_results(args.video_path, foreground_masks_dict, args.output_dir, sky=False)
    
 

if __name__ == "__main__":
    main() 