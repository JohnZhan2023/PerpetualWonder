import os
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def images_to_mp4(images_dir, output_mp4, fps=30, size=None):
    # Get all image files and sort them
    images = [img for img in os.listdir(images_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()
    if not images:
        raise ValueError("No images found in images directory!")
    # Read first image to determine resolution
    first_image = cv2.imread(os.path.join(images_dir, images[0]))
    if size is None:
        h, w = first_image.shape[:2]
        size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, size)
    for img_name in images:
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, size)
        out.write(img)
    out.release()
    print(f"Video saved to: {output_mp4}")