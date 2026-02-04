"""
Physics simulation script based on Genesis.

This script performs physics-based simulations for 3D scenes. The simulation 
configuration is defined in simulation_config.py in the same directory. You can 
control which round of simulation to run by passing the round_num parameter.

Output:
    Running this script generates the following files and directories under 
    3d_result/{scene_name}/stage2_forwardpass/:
    - simulation/: Directory containing simulation output videos
    - mesh_output/: Directory containing mesh output files
    - physical_params/: Directory containing physical parameters for each frame
"""

import taichi as ti
import torch
import pickle
import yaml
import math
import sys
sys.path.append("./")
from toolkit.Scene_3d import Scene_3d
import genesis as gs
from simulation_config import config_scene_3d
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_physical_params(phys_dir: str, frame_id: int) -> torch.Tensor:
    dir_list = os.listdir(phys_dir)
    dir_list.sort()
    params_list = []
    for dir in dir_list:
        params_path = os.path.join(phys_dir, dir, f"{frame_id:04d}.npy")
        params = np.load(params_path)
        params = torch.from_numpy(params).float()
        params_list.append(params)
    return params_list

def load_custom_particles(config, round_num):
    """
    Load custom particles from a pt file
    Find the pt file starting with generative_simulation and sort
    Return a list [dict, dict, dict, ...]
    """
    pt_file = os.path.join(config['work_dir'], "stage3_optimization", "generative_simulation", f'round_{round_num-1}.pt')

    result = torch.load(pt_file)
    return result

def main(config_path, round_num):
    config = load_config(config_path)
    gs.init(backend=gs.cuda)
    if round_num > 1:
        generative_simulation = load_custom_particles(config, round_num)
        phys_dir = os.path.join(config['work_dir'], 'stage2_forwardpass', 'physical_params', f"round_{round_num-1}")
        particles_info = load_physical_params(phys_dir, 384)
    else:
        generative_simulation = None
    scene_3d = Scene_3d(work_dir=config['work_dir'], bottom_threshold=config['simulator_config']['bottom_threshold'], round_num=round_num, config=config)
    if 'vis_frequency' in config['simulator_config']:
        vis_frequency = config['simulator_config']['vis_frequency']
    else:
        vis_frequency = 8

    scene = config_scene_3d(config, scene_3d)
    if 'camera_pos' in config['simulator_config']:
        camera_pos = config['simulator_config']['camera_pos']
    else:
        camera_pos = (-3.0, -1.0, 0.5)
    cam = scene.add_camera(
        res    = (512, 512),
        pos    = camera_pos,
        lookat = (0, 0, 0.0),
        fov    = 60,
        GUI    = False
    )

    scene.build()

    entities = scene.entities

    simulation_entity_id = config['simulator_config']['simulation_entity_id']
    entities = [entities[i] for i in simulation_entity_id]

    if generative_simulation is not None and round_num > 1:
        print(f"============ loading config from round{round_num-1} ============")
        for i, generative_simulation_i in enumerate(generative_simulation):

            if not config["simulator_config"]["entities"][i]["is_rigid_body"]:
                print(f"current entity {i} is not rigid body")
                xyz = generative_simulation_i['all_frames_xyz']
                if type(xyz) == list:
                    xyz = xyz[-1]

                entities[i].activate()
                entities[i].set_position(xyz + torch.tensor([0, 0, -config['simulator_config']['lift_height']]).to("cuda"))
                entities[i].set_velocity(particles_info[i][1])
            else:
                print(f"current entity {i} is rigid body")
                result = np.load(os.path.join(phys_dir, f"entity_{i}", "qpos.npy"))
                qpos_initial = result[:7]
                vel_6d = result[7:]
                p_add = generative_simulation_i['rigid_translations'][-1].cpu().detach().numpy()
                q_add_wxyz = generative_simulation_i['rigid_rotations'][-1].cpu().detach().numpy()
                q_add_xyzw = q_add_wxyz[[1, 2, 3, 0]]

                p_initial = qpos_initial[:3]
                q_initial_wxyz = qpos_initial[3:]
                q_initial_xyzw = q_initial_wxyz[[1, 2, 3, 0]]

                initial_rot = R.from_quat(q_initial_xyzw)
                add_rot = R.from_quat(q_add_xyzw)

                final_rot = add_rot * initial_rot
                final_p = p_initial + initial_rot.apply(p_add)

                qpos_final = np.zeros(7)
                qpos_final[:3] = final_p
                final_q_xyzw = final_rot.as_quat()
                qpos_final[3:] = final_q_xyzw[[3, 0, 1, 2]]

                entities[i].set_qpos(qpos_final)

                entities[i].set_dofs_velocity(vel_6d)
        scene.step()

    cam.start_recording()
    camera_list = config['simulator_config']['camera_list']
    scene_3d.load_camera_params(camera_list)

    T = config['simulator_config']['T']

    for i in range(T):
        if config['scene_name'] == "dumpling" and round_num == 2:

            entity = entities[1]
            if i <= 60:
                target_dofs = np.array([ 0.1168, -3.3334, -0.0077, -0.9911, -0.2164, -0.2139])
                entity.control_dofs_position(target_dofs, None)
            elif i >= 60:
                target_dofs = np.array([ 3.1168, -0.5334, -3.0077, -1.3911, -0.2164, -0.2139])
                entity.control_dofs_position(target_dofs, None)

        elif config['scene_name'] == "dumpling" and round_num == 3:
            entity = entities[1]
            target_dofs = np.array([ -5.5399, -0.7318, -5.0670, -1.5890, -0.5069,  -0.4069])
            entity.control_dofs_position(target_dofs, None)

        ti.sync()
        if i % vis_frequency == 0:
            render_out = cam.render()
            for n, entity in enumerate(entities):
                scene_3d.update(i, entity, n)
        scene.step()

    cam.stop_recording(save_to_filename = scene_3d.simulation_dir + f'/object_{round_num}.mp4', fps = 24)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--round_num', default=1, type=int, help='Round number')
    args = parser.parse_args()
    main(args.config, args.round_num)
