import torch
import numpy as np
import imageio
import os
from gsplat.rendering import rasterization_2dgs, rasterization

"""
Mixed rendering module: supports foreground 3DGS + background 2DGS rendering

Main functions:
1. render_gs: Original 2DGS rendering function
2. render_mixed_gs: New mixed rendering function, foreground uses 3DGS, background uses 2DGS

Usage:
# Prepare foreground and background Gaussian data
foreground_data = {
    'means': fg_means,    # Foreground Gaussian centers
    'quats': fg_quats,    # Foreground Gaussian rotations
    'scales': fg_scales,  # Foreground Gaussian scales
    'opacities': fg_opacities,  # Foreground Gaussian opacities
    'colors': fg_colors   # Foreground Gaussian colors
}

background_data = {
    'means': bg_means,    # Background Gaussian centers
    'quats': bg_quats,    # Background Gaussian rotations
    'scales': bg_scales,  # Background Gaussian scales
    'opacities': bg_opacities,  # Background Gaussian opacities
    'colors': bg_colors   # Background Gaussian colors
}

# Call mixed rendering
render_colors, render_alphas = render_mixed_gs(cam, foreground_data, background_data, background_color)
"""


def render_gs(cam, means, quats, scales, opacities, colors, background_color):
    """Render 3D Gaussian splats"""
    # Ensure inputs are float32 type
    worldtocam = torch.linalg.inv(cam.camtoworld).float()
    K = cam.K.float()
    # Ensure on the correct device
    worldtocam = worldtocam.to(cam.device)
    K = cam.K.to(cam.device)

    # unsqueeze the first dimension
    worldtocam = worldtocam.unsqueeze(0)
    K = K.unsqueeze(0)

    # 1. Get Gaussian splat data and ensure all tensors are on the correct device
    means = means.float().to(cam.device)  # [N, 3] 
    quats = quats.float().to(cam.device)  # [N, 4]
    scales = scales.float().to(cam.device)  # [N, 3]
    opacities = opacities.float().to(cam.device)  # [N]
    
    # 2. Get color data and normalize to [0,1] range
    colors = colors.float().to(cam.device)  # [N, K, 3]

    # 4. Ensure all tensors are on the same device
    assert means.device == quats.device == scales.device == opacities.device == colors.device == worldtocam.device == K.device, f"Device mismatch: means={means.device}, quats={quats.device}, scales={scales.device}, opacities={opacities.device}, colors={colors.device}, worldtocam={worldtocam.device}, K={K.device}"
    
    # Ensure width and height are int type
    width = int(cam.width)
    height = int(cam.height)
    
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
        viewmats=worldtocam,
        Ks=K,
        width=width,
        height=height,
        packed=False,
        absgrad=False,
        sparse_grad=False,
        sh_degree = 0,
        render_mode="RGB+ED",
        near_plane=0.2,
        far_plane=200.0,
        radius_clip=0.0,
        backgrounds=torch.tensor(background_color, dtype=torch.float32).to(cam.device).view(1, 3),
    )
    return render_colors, render_alphas


def render_2d_gs(cam, foreground_data, background_data, background_color):
    """
    Render foreground and background data using 2DGS
    
    Args:
        cam: Camera object
        foreground_data: Foreground Gaussian data dictionary containing means, quats, scales, opacities, colors
        background_data: Background Gaussian data dictionary containing means, quats, scales, opacities, colors  
        background_color: Background color [R, G, B]
    
    Returns:
        render_colors: Rendered color image
        render_alphas: Rendered alpha channel
    """
    # Ensure inputs are float32 type
    worldtocam = torch.linalg.inv(cam.camtoworld).float()
    K = cam.K.float()
    # Ensure on the correct device
    worldtocam = worldtocam.to(cam.device)
    K = cam.K.to(cam.device)

    # unsqueeze the first dimension
    worldtocam = worldtocam.unsqueeze(0)
    K = K.unsqueeze(0)

    # Ensure width and height are int type
    width = int(cam.width)
    height = int(cam.height)
    
    # Prepare merged data lists
    all_means = []
    all_quats = []
    all_scales = []
    all_opacities = []
    all_colors = []

    # 1. Process background data
    if background_data is not None and len(background_data['means']) > 0:
        bg_means = background_data['means'].float().to(cam.device)  # [N, 3]
        bg_quats = background_data['quats'].float().to(cam.device)  # [N, 4]
        bg_scales = background_data['scales'].float().to(cam.device)  # [N, 3]
        bg_opacities = background_data['opacities'].float().to(cam.device)  # [N]
        bg_colors = background_data['colors'].float().to(cam.device)  # [N, K, 3]
        
        # Ensure bg_colors has shape [N, K, 3]
        if len(bg_colors.shape) == 2:  # [N, 3]
            bg_colors = bg_colors.unsqueeze(1)  # [N, 1, 3]
        elif len(bg_colors.shape) > 3:  # [N, 1, 1, 3] or more
            bg_colors = bg_colors.squeeze()  # Remove all dimensions of size 1
            if len(bg_colors.shape) == 2:  # [N, 3]
                bg_colors = bg_colors.unsqueeze(1)  # [N, 1, 3]
        
        all_means.append(bg_means)
        all_quats.append(bg_quats)
        all_scales.append(bg_scales)
        all_opacities.append(bg_opacities)
        all_colors.append(bg_colors)

    # 2. Process foreground data
    if foreground_data is not None and len(foreground_data['means']) > 0:
        fg_means = foreground_data['means'].float().to(cam.device)  # [N, 3]
        fg_quats = foreground_data['quats'].float().to(cam.device)  # [N, 4]
        fg_scales = foreground_data['scales'].float().to(cam.device)  # [N, 3]
        fg_opacities = foreground_data['opacities'].float().to(cam.device)  # [N]
        fg_colors = foreground_data['colors'].float().to(cam.device)  # [N, K, 3]
        
        # Ensure fg_colors has shape [N, K, 3]
        if len(fg_colors.shape) == 2:  # [N, 3]
            fg_colors = fg_colors.unsqueeze(1)  # [N, 1, 3]
        elif len(fg_colors.shape) > 3:  # [N, 1, 1, 3] or more
            fg_colors = fg_colors.squeeze()  # Remove all dimensions of size 1
            if len(fg_colors.shape) == 2:  # [N, 3]
                fg_colors = fg_colors.unsqueeze(1)  # [N, 1, 3]
        
        all_means.append(fg_means)
        all_quats.append(fg_quats)
        all_scales.append(fg_scales)
        all_opacities.append(fg_opacities)
        all_colors.append(fg_colors)

    # 3. If no data, return solid color background
    if not all_means:
        bg_color_tensor = torch.tensor(background_color, dtype=torch.float32, device=cam.device)
        final_colors = bg_color_tensor.view(1, 1, 1, 3).expand(1, height, width, 3)
        final_alpha = torch.zeros(1, height, width, 1, device=cam.device)
        return final_colors, final_alpha

    # 4. Concatenate all data
    means = torch.cat(all_means, dim=0)
    quats = torch.cat(all_quats, dim=0)
    scales = torch.cat(all_scales, dim=0)
    opacities = torch.cat(all_opacities, dim=0)
    colors = torch.cat(all_colors, dim=0)

    # 5. Ensure all tensors are on the same device
    assert means.device == quats.device == scales.device == opacities.device == colors.device == worldtocam.device == K.device, f"Device mismatch: means={means.device}, quats={quats.device}, scales={scales.device}, opacities={opacities.device}, colors={colors.device}, worldtocam={worldtocam.device}, K={K.device}"

    # 6. Render using 2DGS
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
        viewmats=worldtocam,
        Ks=K,
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
        backgrounds=torch.tensor(background_color, dtype=torch.float32).to(cam.device).view(1, 3),
    )
    
    return render_colors, render_alphas


def render_mixed_gs(cam, foreground_data=None, background_data=None, background_color=None):
    """
    Mixed rendering: foreground uses 3DGS, background uses 2DGS
    """
    # Ensure all inputs are on the correct device
    worldtocam = cam.worldtocam.float().to(cam.device)
    K = cam.K.float().to(cam.device)
    
    # gsplat requires [N, 4, 4] and [N, 3, 3] shapes, where N is batch size
    worldtocam = worldtocam.unsqueeze(0)  # [1, 4, 4]
    K = K.unsqueeze(0)  # [1, 3, 3]
    
    # Set background color
    if background_color is None:
        background_color = [0, 0, 0]
    bg_color_tensor = torch.tensor(background_color, dtype=torch.float32, device=cam.device).view(1, 3)
    
    # Ensure width and height are int type
    width = int(cam.width)
    height = int(cam.height)
    
    # 1. Render background (using 2DGS)
    if background_data is not None and len(background_data['means']) > 0:
        bg_means = background_data['means'].float().to(cam.device)  # [N, 3]
        bg_quats = background_data['quats'].float().to(cam.device)  # [N, 4]
        bg_scales = background_data['scales'].float().to(cam.device)  # [N, 3]
        bg_opacities = background_data['opacities'].float().to(cam.device)  # [N]
        bg_colors = background_data['colors'].float().to(cam.device)  # [N, 3]
        
        # Ensure bg_colors has shape [N, K, 3]
        if len(bg_colors.shape) == 2:  # [N, 3]
            bg_colors = bg_colors.unsqueeze(1)  # [N, 1, 3]
        elif len(bg_colors.shape) > 3:  # [N, 1, 1, 3] or more
            bg_colors = bg_colors.squeeze()  # Remove all dimensions of size 1
            if len(bg_colors.shape) == 2:  # [N, 3]
                bg_colors = bg_colors.unsqueeze(1)  # [N, 1, 3]

        # Check that all tensor shapes are correct
        assert len(bg_colors.shape) == 3 and bg_colors.shape[-1] == 3, f"bg_colors shape should be [N, K, 3], got {bg_colors.shape}"
        
        (
            bg_render_colors,
            bg_render_alphas,
            bg_render_normals,
            bg_normals_from_depth,
            bg_render_distort,
            bg_render_median,
            bg_info,
        ) = rasterization_2dgs(
            means=bg_means,
            quats=bg_quats,
            scales=bg_scales,
            opacities=bg_opacities,
            colors=bg_colors,
            viewmats=worldtocam,
            Ks=K,
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
            backgrounds=bg_color_tensor,
        )
    else:
        # If no background Gaussians, use solid color background
        bg_render_colors = bg_color_tensor.view(1, 1, 1, 3).expand(1, cam.height, cam.width, 3)
        bg_render_alphas = torch.zeros(1, cam.height, cam.width, 1, device=cam.device)  # Set to 0 so background is fully transparent
    
    # 2. Render foreground (using 3DGS)
    if foreground_data is not None and len(foreground_data['means']) > 0:

        fg_means = foreground_data['means'].float().to(cam.device)
        fg_quats = foreground_data['quats'].float().to(cam.device)
        fg_scales = foreground_data['scales'].float().to(cam.device)
        fg_opacities = foreground_data['opacities'].float().to(cam.device)
        fg_colors = foreground_data['colors'].float().to(cam.device)
        
        # For 3DGS, use transparent background for later compositing
        transparent_bg = torch.zeros(1, 3, device=cam.device)
        
        fg_render_colors, fg_render_alphas, fg_info = rasterization(
            means=fg_means,
            quats=fg_quats,
            scales=fg_scales,
            opacities=fg_opacities,
            colors=fg_colors,
            viewmats=worldtocam,
            Ks=K,
            width=cam.width,
            height=cam.height,
            packed=False,
            sparse_grad=False,
            absgrad=False,
            sh_degree=0,
            render_mode="RGB+ED",
            near_plane=0.2,
            far_plane=200.0,
            radius_clip=0.0,
            backgrounds=transparent_bg,
        )
    else:
        # If no foreground Gaussians, create transparent foreground
        fg_render_colors = torch.zeros(1, cam.height, cam.width, 4, device=cam.device)
        fg_render_alphas = torch.zeros(1, cam.height, cam.width, 1, device=cam.device)
    
    # 3. Composite foreground and background
    # Extract RGB channels
    if fg_render_colors.shape[-1] >= 3:
        fg_rgb = fg_render_colors[..., :3]  # [1, H, W, 3]
    else:
        fg_rgb = torch.zeros(1, cam.height, cam.width, 3, device=cam.device)
        
    if bg_render_colors.shape[-1] >= 3:
        bg_rgb = bg_render_colors[..., :3]  # [1, H, W, 3]
    else:
        bg_rgb = bg_color_tensor.view(1, 1, 1, 3).expand(1, cam.height, cam.width, 3)
    
    # Alpha compositing: fg_rgb * fg_alpha + bg_rgb * (1 - fg_alpha)
    fg_alpha = fg_render_alphas  # [1, H, W, 1]
    final_rgb = fg_rgb * fg_alpha + bg_rgb * (1 - fg_alpha)
    # clamp the final_rgb to [0, 1]
    final_rgb = torch.clamp(final_rgb, 0, 1)
    
    # Composite alpha: fg_alpha + bg_alpha * (1 - fg_alpha)
    final_alpha = fg_alpha + bg_render_alphas * (1 - fg_alpha)
    
    # If depth information is needed, can similarly composite depth
    if fg_render_colors.shape[-1] == 4 and bg_render_colors.shape[-1] == 4:
        fg_depth = fg_render_colors[..., 3:4]
        bg_depth = bg_render_colors[..., 3:4]
        final_depth = fg_depth * fg_alpha + bg_depth * (1 - fg_alpha)
        final_colors = torch.cat([final_rgb, final_depth], dim=-1)
    else:
        final_colors = final_rgb
    
    return final_colors, final_alpha


def render_concat_gs(cam, foreground_data=None, background_data=None, background_color=None):
    """
    Concatenation rendering: concatenate foreground and background attributes, then use unified 3DGS rendering pipeline
    """
    # Ensure all inputs are on the correct device
    worldtocam = cam.worldtocam.float().to(cam.device)
    K = cam.K.float().to(cam.device)
    
    # gsplat requires [N, 4, 4] and [N, 3, 3] shapes, where N is batch size
    worldtocam = worldtocam.unsqueeze(0)  # [1, 4, 4]
    K = K.unsqueeze(0)  # [1, 3, 3]
    
    # Set background color
    if background_color is None:
        background_color = [0, 0, 0]
    bg_color_tensor = torch.tensor(background_color, dtype=torch.float32, device=cam.device).view(1, 3)
    
    # Ensure width and height are int type
    width = int(cam.width)
    height = int(cam.height)
    
    # Prepare foreground and background data
    all_means = []
    all_quats = []
    all_scales = []
    all_opacities = []
    all_colors = []

    # 1. Collect foreground data
    if foreground_data is not None and len(foreground_data['means']) > 0:
        all_means.append(foreground_data['means'].float().to(cam.device))
        all_quats.append(foreground_data['quats'].float().to(cam.device))
        all_scales.append(foreground_data['scales'].float().to(cam.device))
        all_opacities.append(foreground_data['opacities'].float().to(cam.device))
        all_colors.append(foreground_data['colors'].float().to(cam.device))

    # 2. Collect background data
    if background_data is not None and len(background_data['means']) > 0:
        all_means.append(background_data['means'].float().to(cam.device))
        all_quats.append(background_data['quats'].float().to(cam.device))
        all_scales.append(background_data['scales'].float().to(cam.device))
        all_opacities.append(background_data['opacities'].float().to(cam.device).squeeze())
        all_colors.append(background_data['colors'].float().to(cam.device))

    # 3. If no data, return solid color background
    if not all_means:
        final_colors = bg_color_tensor.view(1, 1, 1, 3).expand(1, height, width, 3)
        final_alpha = torch.zeros(1, height, width, 1, device=cam.device)
        return final_colors, final_alpha

    # 4. Concatenate all data
    means = torch.cat(all_means, dim=0)
    quats = torch.cat(all_quats, dim=0)
    scales = torch.cat(all_scales, dim=0)
    opacities = torch.cat(all_opacities, dim=0)
    colors = torch.cat(all_colors, dim=0)

    # Ensure colors has shape [N, K, 3]
    if len(colors.shape) == 2:  # [N, 3]
        colors = colors.unsqueeze(1)  # [N, 1, 3]
    elif len(colors.shape) > 3:  # [N, 1, 1, 3] or more
        colors = colors.squeeze()  # Remove all dimensions of size 1
        if len(colors.shape) == 2:  # [N, 3]
            colors = colors.unsqueeze(1)  # [N, 1, 3]

    # 5. Render using 3DGS rendering pipeline
    render_colors, render_alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=worldtocam,
        Ks=K,
        width=width,
        height=height,
        packed=False,
        sparse_grad=False,
        absgrad=False,
        sh_degree=0,
        render_mode="RGB+ED",
        near_plane=0.2,
        far_plane=200.0,
        radius_clip=0.0,
        backgrounds=bg_color_tensor,
    )

    return render_colors, render_alpha

