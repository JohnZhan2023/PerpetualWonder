#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from depth_diff_gaussian_rasterization_min import GaussianRasterizationSettings, GaussianRasterizer
# this is the own customed in HUGS paper
from diff_gaussian_rasterization import GaussianRasterizer as GaussianRasterizer_hugs
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_4d import GaussianModel as GaussianModel_4d


def proj_uv(xyz, cam):
    device = xyz.device
    intr = torch.eye(3).float().to(device)
    intr[0, 0] = cam.focal_x
    intr[1, 1] = cam.focal_y
    intr[0, 2] = cam.image_width / 2.0
    intr[1, 2] = cam.image_height / 2.0
    w2c = torch.eye(4).float().to(device)
    w2c[:3, :3] = torch.tensor(cam.R.T).to(device)
    w2c[:3, 3] = torch.tensor(cam.T).to(device)

    c_xyz = (w2c[:3, :3] @ xyz.T).T + w2c[:3, 3]
    i_xyz = (intr @ c_xyz.mT).mT  # (N, 3)
    uv = i_xyz[:, :2] / i_xyz[:, -1:].clip(1e-3) # (N, 2)
    return uv


def render_w_shit_flow(
    viewpoint_camera,
    sim_out_step,
    pc: GaussianModel, 
    opt, 
    bg_color: torch.Tensor, 
    scaling_modifier=1.0, 
    override_color=None, 
    render_visible=False, 
    exclude_sky=False,
    render_mask=None,
    object_pts_num_list=None,
    prev_step=None,
    prev_step_cam=None,
):
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    full_proj_matrix = viewpoint_camera.full_proj_transform
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=full_proj_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=True
    )

    rasterizer = GaussianRasterizer_hugs(raster_settings=raster_settings)

    obj_xyz = []
    obj_rotation = []
    obj_scaling = []
    obj_features_dc = []
    obj_opacity = []
    obj_valid_masks = []
    obj_nums = []

    for k, v in sim_out_step.items():
        obj_xyz.append(v['xyz'])
        obj_rotation.append(v['rotation'])
        obj_scaling.append(v['scaling'])
        obj_features_dc.append(v['features_dc'])
        obj_opacity.append(v['opacity'])
        obj_valid_masks.append(v['valid_mask'])
        obj_nums.append(v['xyz'].shape[0])
    
    obj_xyz = torch.cat(obj_xyz, dim=0)
    obj_rotation = torch.cat(obj_rotation, dim=0)
    obj_scaling = torch.cat(obj_scaling, dim=0)
    obj_features_dc = torch.cat(obj_features_dc, dim=0)
    obj_opacity = torch.cat(obj_opacity, dim=0)

    with torch.no_grad():
        _, xyz_prev = pc._tmp_get_xyz_all_separate()
    means3D = torch.cat([obj_xyz, xyz_prev], dim=0)

    if prev_step is not None:
        prev_obj_xyz = []
        for k,v in prev_step.items():
            prev_obj_xyz.append(v['xyz'])
        prev_obj_xyz = torch.cat(prev_obj_xyz, dim=0)

        prev_viewpoint_camera = prev_step_cam

        uv = proj_uv(obj_xyz, viewpoint_camera)
        prev_uv = proj_uv(prev_obj_xyz, prev_viewpoint_camera)

        if uv.shape[0] > prev_uv.shape[0]:
            # because of emitter, points number is different
            prev_uv_more = uv[-(uv.shape[0] - prev_uv.shape[0]):]
            prev_uv = torch.cat([prev_uv, prev_uv_more], dim=0)
        
        delta_uv = uv - prev_uv
        
        # the foreground we visualize the 3rd channel also as 0
        delta_uv = torch.cat([delta_uv, torch.zeros_like(delta_uv[:, :1], device=delta_uv.device)], dim=-1)
        # the sky point cloud's closest z should be 0.02 * cos(min_latitude) * cos(longitude_offset) = 0.01511
        uv_env = proj_uv(xyz_prev, viewpoint_camera)
        uv_env_prev = proj_uv(xyz_prev, prev_viewpoint_camera)
        delta_uv_env = uv_env - uv_env_prev
        # some points in background are numerically unstable, so we set them to 0 if they are just outside of the image
        delta_uv_invalid_image = (uv_env < 0).any(dim=1) | (uv_env > 512).any(dim=1) | (uv_env_prev < 0).any(dim=1) | (uv_env_prev > 512).any(dim=1)
        delta_uv_invalid_sky = xyz_prev[:, 2] >= (0.01511 * 1000)
        delta_uv_invalid = torch.logical_or(delta_uv_invalid_image, delta_uv_invalid_sky)
        delta_uv_env[delta_uv_invalid] = 0
        delta_uv_env = torch.cat([delta_uv_env, torch.zeros_like(delta_uv_env[:, :1], device=delta_uv_env.device)], dim=-1)
        
        delta_uv_all = torch.cat([delta_uv, delta_uv_env], dim=0)
    else:
        delta_uv_all = torch.zeros_like(means3D)

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # everything from original Gaussians are with disabled gradient
    with torch.no_grad():
        means2D = screenspace_points
        
        # opacity = pc.get_opacity_all
        _, opacity_prev = pc._tmp_get_opacity_all_separate()
        opacity = torch.cat([obj_opacity, opacity_prev], dim=0)

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if opt.compute_cov3D_python:
            # cov3D_precomp = pc.get_covariance(scaling_modifier)
            cov3D_precomp = pc.get_covariance_all(scaling_modifier)
        else:
            # scales = pc.get_scaling_with_3D_filter
            # rotations = pc.get_rotation
            # scales = pc.get_scaling_with_3D_filter_all

            # scales = pc.get_scaling_all
            # rotations = pc.get_rotation_all
            _, scales_prev = pc._tmp_get_scaling_all_separate()
            scales = torch.cat([obj_scaling, scales_prev], dim=0)
            _, rotations_prev = pc._tmp_get_rotation_all_separate()
            rotations = torch.cat([obj_rotation, rotations_prev], dim=0)
        
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if opt.convert_SHs_python:
                # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                # shs_view = pc.get_features_all.transpose(1, 2).view(-1, 3)
                _, shs_prev = pc._tmp_get_features_dc_all_separate()
                shs_view = torch.cat([obj_features_dc, shs_prev], dim=0)

                # dir_pp = (pc.get_xyz_all - viewpoint_camera.camera_center.repeat(pc.get_features_all.shape[0], 1))
                # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                colors_precomp = pc.color_activation(shs_view)
            else:
                # shs = pc.get_features
                shs = pc.get_features_all
        else:
            colors_precomp = override_color

        if render_visible:
            visibility_filter_all = pc.visibility_filter_all  # Seen in screen
        else:
            visibility_filter_all = torch.ones_like(pc.visibility_filter_all, dtype=torch.bool)

        if exclude_sky:
            visibility_filter_all = visibility_filter_all & ~pc.is_sky_filter
        else:
            visibility_filter_all = visibility_filter_all
    
    # rearrange the shape of visibility_filter and render_mask here, since objects particle numbers maybe different
    visibility_filter_env = visibility_filter_all[-xyz_prev.shape[0]:]
    visibility_filter_obj = []
    for iid, num in enumerate(obj_nums):
        this_visibility = torch.ones(num).bool().to(visibility_filter_env.device)
        visibility_filter_obj.append(this_visibility)
    visibility_filter_obj = torch.cat(visibility_filter_obj, dim=0)
    visibility_filter_all = torch.cat([visibility_filter_obj, visibility_filter_env], dim=0)

    if render_mask is not None:
        render_mask_env = render_mask[-xyz_prev.shape[0]:]
        render_mask_obj = []
        for iid, num in enumerate(obj_nums):
            this_render_mask = torch.ones(num).bool().to(render_mask_env.device)
            render_mask_obj.append(this_render_mask)
        render_mask_obj = torch.cat(render_mask_obj, dim=0)
        render_mask = torch.cat([render_mask_obj, render_mask_env], dim=0)
        render_mask = torch.logical_and(render_mask, visibility_filter_all)

    means3D = means3D[visibility_filter_all]
    means2D = means2D[visibility_filter_all]
    shs = None if shs is None else shs[visibility_filter_all]
    colors_precomp = None if colors_precomp is None else colors_precomp[visibility_filter_all]
    opacity = opacity[visibility_filter_all]
    scales = scales[visibility_filter_all]
    rotations = rotations[visibility_filter_all]
    cov3D_precomp = None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    feats3D = torch.zeros(means3D.shape[0], 20).to(means3D.device)
    rendered_image, _, feats, depth, flow = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None, # shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        feats3D = feats3D,
        delta = delta_uv_all)
    del feats
    ret = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            # "final_opacity": torch.ones_like(rendered_image),
            "depth": depth,
            "median_depth": None,
            "optical_flow": flow}

    return ret


def render_w_shit_da(
    viewpoint_camera,
    sim_out_step,
    pc: GaussianModel, 
    opt, 
    bg_color: torch.Tensor, 
    scaling_modifier=1.0, 
    override_color=None, 
    render_visible=False, 
    exclude_sky=False,
    render_mask=None,
    object_pts_num_list=None,
):
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    full_proj_matrix = viewpoint_camera.full_proj_transform
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=full_proj_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=opt.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    obj_xyz = []
    obj_rotation = []
    obj_scaling = []
    obj_features_dc = []
    obj_opacity = []
    obj_valid_masks = []
    obj_nums = []

    for k, v in sim_out_step.items():
        obj_xyz.append(v['xyz'])
        obj_rotation.append(v['rotation'])
        obj_scaling.append(v['scaling'])
        obj_features_dc.append(v['features_dc'])
        obj_opacity.append(v['opacity'])
        obj_valid_masks.append(v['valid_mask'])
        obj_nums.append(v['xyz'].shape[0])
    
    obj_xyz = torch.cat(obj_xyz, dim=0)
    obj_rotation = torch.cat(obj_rotation, dim=0)
    obj_scaling = torch.cat(obj_scaling, dim=0)
    obj_features_dc = torch.cat(obj_features_dc, dim=0)
    obj_opacity = torch.cat(obj_opacity, dim=0)

    with torch.no_grad():
        _, xyz_prev = pc._tmp_get_xyz_all_separate()
    means3D = torch.cat([obj_xyz, xyz_prev], dim=0)

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # everything from original Gaussians are with disabled gradient
    with torch.no_grad():
        means2D = screenspace_points
        
        # opacity = pc.get_opacity_all
        _, opacity_prev = pc._tmp_get_opacity_all_separate()
        opacity = torch.cat([obj_opacity, opacity_prev], dim=0)

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if opt.compute_cov3D_python:
            # cov3D_precomp = pc.get_covariance(scaling_modifier)
            cov3D_precomp = pc.get_covariance_all(scaling_modifier)
        else:
            # scales = pc.get_scaling_with_3D_filter
            # rotations = pc.get_rotation
            # scales = pc.get_scaling_with_3D_filter_all

            # scales = pc.get_scaling_all
            # rotations = pc.get_rotation_all
            _, scales_prev = pc._tmp_get_scaling_all_separate()
            scales = torch.cat([obj_scaling, scales_prev], dim=0)
            _, rotations_prev = pc._tmp_get_rotation_all_separate()
            rotations = torch.cat([obj_rotation, rotations_prev], dim=0)
        
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if opt.convert_SHs_python:
                # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                # shs_view = pc.get_features_all.transpose(1, 2).view(-1, 3)
                _, shs_prev = pc._tmp_get_features_dc_all_separate()
                shs_view = torch.cat([obj_features_dc, shs_prev], dim=0)

                # dir_pp = (pc.get_xyz_all - viewpoint_camera.camera_center.repeat(pc.get_features_all.shape[0], 1))
                # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                colors_precomp = pc.color_activation(shs_view)
            else:
                # shs = pc.get_features
                shs = pc.get_features_all
        else:
            colors_precomp = override_color

        if render_visible:
            visibility_filter_all = pc.visibility_filter_all  # Seen in screen
        else:
            visibility_filter_all = torch.ones_like(pc.visibility_filter_all, dtype=torch.bool)

        if exclude_sky:
            visibility_filter_all = visibility_filter_all & ~pc.is_sky_filter
        else:
            visibility_filter_all = visibility_filter_all
    
    # rearrange the shape of visibility_filter and render_mask here, since objects particle numbers maybe different
    visibility_filter_env = visibility_filter_all[-xyz_prev.shape[0]:]
    visibility_filter_obj = []
    for iid, num in enumerate(obj_nums):
        this_visibility = torch.ones(num).bool().to(visibility_filter_env.device)
        visibility_filter_obj.append(this_visibility)
    visibility_filter_obj = torch.cat(visibility_filter_obj, dim=0)
    visibility_filter_all = torch.cat([visibility_filter_obj, visibility_filter_env], dim=0)

    if render_mask is not None:
        # in for da, there will not be new object in passed in render_mask
        render_mask_env = render_mask[-xyz_prev.shape[0]:]
        render_mask_obj = []
        start_id = 0
        for iid, num in enumerate(obj_nums):
            this_render_mask = render_mask[start_id:start_id+num]
            render_mask_obj.append(this_render_mask)
            start_id += num
        render_mask_obj = torch.cat(render_mask_obj, dim=0)
        render_mask = torch.cat([render_mask_obj, render_mask_env], dim=0)
        # render_mask = torch.logical_and(render_mask, visibility_filter_all)
        visibility_filter_all = torch.logical_and(visibility_filter_all, render_mask)

    means3D = means3D[visibility_filter_all]
    means2D = means2D[visibility_filter_all]
    shs = None if shs is None else shs[visibility_filter_all]
    colors_precomp = None if colors_precomp is None else colors_precomp[visibility_filter_all]
    opacity = opacity[visibility_filter_all]
    scales = scales[visibility_filter_all]
    rotations = rotations[visibility_filter_all]
    cov3D_precomp = None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, median_depth, final_opacity = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,)
    
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "final_opacity": final_opacity,
            "depth": depth,
            "median_depth": median_depth,}


def render_w_shit(
    viewpoint_camera,
    sim_out_step,
    pc: GaussianModel, 
    opt, 
    bg_color: torch.Tensor, 
    scaling_modifier=1.0, 
    override_color=None, 
    render_visible=False, 
    exclude_sky=False,
    render_mask=None,
    object_pts_num_list=None,
):
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    full_proj_matrix = viewpoint_camera.full_proj_transform
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=full_proj_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=opt.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    obj_xyz = []
    obj_rotation = []
    obj_scaling = []
    obj_features_dc = []
    obj_opacity = []
    obj_valid_masks = []
    obj_nums = []

    for k, v in sim_out_step.items():
        obj_xyz.append(v['xyz'])
        obj_rotation.append(v['rotation'])
        obj_scaling.append(v['scaling'])
        obj_features_dc.append(v['features_dc'])
        obj_opacity.append(v['opacity'])
        obj_valid_masks.append(v['valid_mask'])
        obj_nums.append(v['xyz'].shape[0])
    
    obj_xyz = torch.cat(obj_xyz, dim=0)
    obj_rotation = torch.cat(obj_rotation, dim=0)
    obj_scaling = torch.cat(obj_scaling, dim=0)
    obj_features_dc = torch.cat(obj_features_dc, dim=0)
    obj_opacity = torch.cat(obj_opacity, dim=0)

    with torch.no_grad():
        _, xyz_prev = pc._tmp_get_xyz_all_separate()
    means3D = torch.cat([obj_xyz, xyz_prev], dim=0)

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # everything from original Gaussians are with disabled gradient
    with torch.no_grad():
        means2D = screenspace_points
        
        # opacity = pc.get_opacity_all
        _, opacity_prev = pc._tmp_get_opacity_all_separate()
        opacity = torch.cat([obj_opacity, opacity_prev], dim=0)

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if opt.compute_cov3D_python:
            # cov3D_precomp = pc.get_covariance(scaling_modifier)
            cov3D_precomp = pc.get_covariance_all(scaling_modifier)
        else:
            # scales = pc.get_scaling_with_3D_filter
            # rotations = pc.get_rotation
            # scales = pc.get_scaling_with_3D_filter_all

            # scales = pc.get_scaling_all
            # rotations = pc.get_rotation_all
            _, scales_prev = pc._tmp_get_scaling_all_separate()
            scales = torch.cat([obj_scaling, scales_prev], dim=0)
            _, rotations_prev = pc._tmp_get_rotation_all_separate()
            rotations = torch.cat([obj_rotation, rotations_prev], dim=0)
        
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if opt.convert_SHs_python:
                # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                # shs_view = pc.get_features_all.transpose(1, 2).view(-1, 3)
                _, shs_prev = pc._tmp_get_features_dc_all_separate()
                shs_view = torch.cat([obj_features_dc, shs_prev], dim=0)

                # dir_pp = (pc.get_xyz_all - viewpoint_camera.camera_center.repeat(pc.get_features_all.shape[0], 1))
                # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                colors_precomp = pc.color_activation(shs_view)
            else:
                # shs = pc.get_features
                shs = pc.get_features_all
        else:
            colors_precomp = override_color

        if render_visible:
            visibility_filter_all = pc.visibility_filter_all  # Seen in screen
        else:
            visibility_filter_all = torch.ones_like(pc.visibility_filter_all, dtype=torch.bool)

        if exclude_sky:
            visibility_filter_all = visibility_filter_all & ~pc.is_sky_filter
        else:
            visibility_filter_all = visibility_filter_all
    
    # rearrange the shape of visibility_filter and render_mask here, since objects particle numbers maybe different
    visibility_filter_env = visibility_filter_all[-xyz_prev.shape[0]:]
    visibility_filter_obj = []
    for iid, num in enumerate(obj_nums):
        this_visibility = torch.ones(num).bool().to(visibility_filter_env.device)
        visibility_filter_obj.append(this_visibility)
    visibility_filter_obj = torch.cat(visibility_filter_obj, dim=0)
    visibility_filter_all = torch.cat([visibility_filter_obj, visibility_filter_env], dim=0)

    if render_mask is not None:
        render_mask_env = render_mask[-xyz_prev.shape[0]:]
        render_mask_obj = []
        for iid, num in enumerate(obj_nums):
            this_render_mask = torch.ones(num).bool().to(render_mask_env.device)
            render_mask_obj.append(this_render_mask)
        render_mask_obj = torch.cat(render_mask_obj, dim=0)
        render_mask = torch.cat([render_mask_obj, render_mask_env], dim=0)
        # render_mask = torch.logical_and(render_mask, visibility_filter_all)
        visibility_filter_all = torch.logical_and(visibility_filter_all, render_mask)
    
    # # because of operations like particle removing for fluid, 
    # # the shape of visibility_filter_all and render_mask may
    # # don't match the number of gaussians in sim_out_step
    # original_foreground_num = visibility_filter_all.shape[0] - xyz_prev.shape[0]
    # current_foreground_num = obj_xyz.shape[0]
    # if current_foreground_num != original_foreground_num:
    #     assert len(object_pts_num_list) == len(obj_nums)
    #     new_visibility_filter_all = []
    #     start_idx = 0
    #     for iid, (original_num, current_num) in enumerate(zip(object_pts_num_list, obj_nums)):
    #         this_visibility_filter = visibility_filter_all[start_idx:start_idx+original_num]
    #         this_valid_mask = obj_valid_masks[iid]
    #         new_visibility_filter_all.append(this_visibility_filter[this_valid_mask])
    #         start_idx += original_num
    #     new_visibility_filter_all = torch.cat(new_visibility_filter_all, dim=0)
    #     new_visibility_filter_all = torch.cat([new_visibility_filter_all, visibility_filter_all[-xyz_prev.shape[0]:]], dim=0)
    #     visibility_filter_all = new_visibility_filter_all

    #     if render_mask is not None:
    #         new_render_mask = []
    #         start_idx = 0
    #         for iid, original_num in enumerate(object_pts_num_list):
    #             this_render_mask = render_mask[start_idx:start_idx+original_num]
    #             this_valid_mask = obj_valid_masks[iid]
    #             new_render_mask.append(this_render_mask[this_valid_mask])
    #             start_idx += original_num
    #         new_render_mask = torch.cat(new_render_mask, dim=0)
    #         new_render_mask = torch.cat([new_render_mask, render_mask[-xyz_prev.shape[0]:]], dim=0)
    #         render_mask = new_render_mask
    
    # # very hack now, first K are the object gaussians and just concat N times
    # # uncomment this if you're hacking to duplicate objects
    # # total_vis_filter_all = torch.cat([visibility_filter_all[:obj_nums[0]]]*len(obj_nums) + [visibility_filter_all[obj_nums[0]:]], dim=0)
    # # visibility_filter_all = total_vis_filter_all
    # if render_mask is not None:
    #     visibility_filter_all = torch.logical_and(visibility_filter_all, render_mask)
    
    # also uncomment this if you're hacking to duplicate objects
    # total_means2D = torch.cat([means2D[:obj_nums[0]]]*len(obj_nums) + [means2D[obj_nums[0]:]], dim=0)
    # means2D = total_means2D

    means3D = means3D[visibility_filter_all]
    means2D = means2D[visibility_filter_all]
    shs = None if shs is None else shs[visibility_filter_all]
    colors_precomp = None if colors_precomp is None else colors_precomp[visibility_filter_all]
    opacity = opacity[visibility_filter_all]
    scales = scales[visibility_filter_all]
    rotations = rotations[visibility_filter_all]
    cov3D_precomp = None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, median_depth, final_opacity = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,)
    
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "final_opacity": final_opacity,
            "depth": depth,
            "median_depth": median_depth,}


def render_dynamic(
    viewpoint_camera,
    pc: GaussianModel, 
    opt, 
    bg_color: torch.Tensor, 
    scaling_modifier=1.0, 
    override_color=None, 
    render_visible=False, 
    exclude_sky=False,
    render_mask=None,
    dynamic_xyz=None
):
    screenspace_points = torch.zeros_like(pc.get_xyz_all, dtype=pc.get_xyz_all.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    full_proj_matrix = viewpoint_camera.full_proj_transform

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=full_proj_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=opt.debug
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    static_means3D = pc.get_xyz_all
    if dynamic_xyz is not None:
        dynamic_means_3D = torch.tensor(dynamic_xyz, dtype=torch.float, device=static_means3D.device)
        # assume object is always at the beginning
        num_object_gaussians = dynamic_means_3D.shape[0]
        static_means3D[:num_object_gaussians] = dynamic_means_3D
        means3D = static_means3D
    else:
        means3D = static_means3D

    means2D = screenspace_points
    opacity = pc.get_opacity_all
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if opt.compute_cov3D_python:
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
        cov3D_precomp = pc.get_covariance_all(scaling_modifier)
    else:
        # scales = pc.get_scaling_with_3D_filter
        # rotations = pc.get_rotation
        # scales = pc.get_scaling_with_3D_filter_all
        scales = pc.get_scaling_all
        rotations = pc.get_rotation_all

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if opt.convert_SHs_python:
            # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            shs_view = pc.get_features_all.transpose(1, 2).view(-1, 3)
            # dir_pp = (pc.get_xyz_all - viewpoint_camera.camera_center.repeat(pc.get_features_all.shape[0], 1))
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            colors_precomp = pc.color_activation(shs_view)
        else:
            # shs = pc.get_features
            shs = pc.get_features_all
    else:
        colors_precomp = override_color

    if render_visible:
        visibility_filter_all = pc.visibility_filter_all  # Seen in screen
    else:
        visibility_filter_all = torch.ones_like(pc.visibility_filter_all, dtype=torch.bool)

    if exclude_sky:
        visibility_filter_all = visibility_filter_all & ~pc.is_sky_filter
    else:
        visibility_filter_all = visibility_filter_all
    
    if render_mask is not None:
        visibility_filter_all = torch.logical_and(visibility_filter_all, render_mask)

    means3D = means3D[visibility_filter_all]
    means2D = means2D[visibility_filter_all]
    shs = None if shs is None else shs[visibility_filter_all]
    colors_precomp = None if colors_precomp is None else colors_precomp[visibility_filter_all]
    opacity = opacity[visibility_filter_all]
    scales = scales[visibility_filter_all]
    rotations = rotations[visibility_filter_all]
    cov3D_precomp = None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, median_depth, final_opacity = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # R = torch.tensor(viewpoint_camera.R, device=means3D.device, dtype=torch.float32)
    # point_normals_in_world = rotation2normal(rotations)
    # point_normals_in_screen = point_normals_in_world @ R

    # render_normal, _, _, _, _ = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = None,
    #     colors_precomp = point_normals_in_screen,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)
    # render_normal = F.normalize(render_normal, dim = 0)        

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "final_opacity": final_opacity,
            "depth": depth,
            "median_depth": median_depth,}


def render_dynamic_4d_single_frame(
    viewpoint_camera,
    pc: GaussianModel_4d, 
    bg_color: torch.Tensor, 
    scaling_modifier=1.0, 
    override_color=None, 
    render_visible=False, 
    exclude_sky=False,
    render_mask=None,
    dynamic_xyz=None,
    frame_id=None
):
    viewmatrix, full_proj_transform, fovx, fovy, campos = viewpoint_camera.get_graphics_camera()
    height, width = viewpoint_camera.height, viewpoint_camera.width
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)
    bg_color = torch.zeros(3, device="cuda:0", dtype=torch.float32)

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=campos,
        prefiltered=False,
        debug=True
    )
    
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz_at_frame(frame_id)
    # turn to tensor
    means3D = torch.tensor(means3D, dtype=torch.float32, device="cuda")
 
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        means2D.retain_grad()
    except:
        pass
    
    # get optimizable opacity
    opacity = pc.get_opacity_all()


    cov3D_precomp = None

    # get optimizable scaling
    scales = pc.get_scaling_all()

    # get optimizable rotation
    rotations = pc.get_rotation_all()

    # get optimizable features
    shs = pc.get_features_all()

    colors_precomp = None

    # get optimizable opacity
    opacity = pc.get_opacity_all()

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, median_depth, final_opacity = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return rendered_image

def render(viewpoint_camera, pc: GaussianModel, opt, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None, render_visible=False, exclude_sky=False,
           timestep=None, movement_sim=None, render_mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros_like(pc.get_xyz_all, dtype=pc.get_xyz_all.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    full_proj_matrix = viewpoint_camera.full_proj_transform

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=full_proj_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=opt.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if timestep is None:
        means3D = pc.get_xyz_all
    else:
        xyz, xyz_prev = pc._tmp_get_xyz_all_separate()  # here xyz is boat, xyz_prev is all other
        if movement_sim is not None:
            sim_idx = int(100 * timestep) // (100 // len(movement_sim))
            movement = movement_sim[sim_idx]
            # xyz_new = xyz + movement
            xyz_new = movement
            # print(xyz_new[0])
        else:
            movement = torch.zeros_like(xyz, device=xyz.device)
            # movement[:, 0] = xyz[:, 0].mean()  # ad hoc number
            # movement[:, 2] = movement[:, 0] * (-1)
            xyz_new = xyz - movement * timestep
        means3D = torch.cat([xyz_new, xyz_prev], dim=0)

    means2D = screenspace_points
    opacity = pc.get_opacity_all

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if opt.compute_cov3D_python:
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
        cov3D_precomp = pc.get_covariance_all(scaling_modifier)
    else:
        # scales = pc.get_scaling_with_3D_filter
        # rotations = pc.get_rotation
        # scales = pc.get_scaling_with_3D_filter_all
        scales = pc.get_scaling_all
        rotations = pc.get_rotation_all

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if opt.convert_SHs_python:
            # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            shs_view = pc.get_features_all.transpose(1, 2).view(-1, 3)
            # dir_pp = (pc.get_xyz_all - viewpoint_camera.camera_center.repeat(pc.get_features_all.shape[0], 1))
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            colors_precomp = pc.color_activation(shs_view)
        else:
            # shs = pc.get_features
            shs = pc.get_features_all
    else:
        colors_precomp = override_color

    if render_visible:
        visibility_filter_all = pc.visibility_filter_all  # Seen in screen
    else:
        visibility_filter_all = torch.ones_like(pc.visibility_filter_all, dtype=torch.bool)

    if exclude_sky:
        visibility_filter_all = visibility_filter_all & ~pc.is_sky_filter
    else:
        visibility_filter_all = visibility_filter_all
    
    if render_mask is not None:
        visibility_filter_all = torch.logical_and(visibility_filter_all, render_mask)

    means3D = means3D[visibility_filter_all]
    means2D = means2D[visibility_filter_all]
    shs = None if shs is None else shs[visibility_filter_all]
    colors_precomp = None if colors_precomp is None else colors_precomp[visibility_filter_all]
    opacity = opacity[visibility_filter_all]
    scales = scales[visibility_filter_all]
    rotations = rotations[visibility_filter_all]
    cov3D_precomp = None if cov3D_precomp is None else cov3D_precomp[visibility_filter_all]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, median_depth, final_opacity = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # R = torch.tensor(viewpoint_camera.R, device=means3D.device, dtype=torch.float32)
    # point_normals_in_world = rotation2normal(rotations)
    # point_normals_in_screen = point_normals_in_world @ R

    # render_normal, _, _, _, _ = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = None,
    #     colors_precomp = point_normals_in_screen,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)
    # render_normal = F.normalize(render_normal, dim = 0)        

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "final_opacity": final_opacity,
            "depth": depth,
            "median_depth": median_depth,}


