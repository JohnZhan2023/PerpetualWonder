import torch
import os
from toolkit.gaussian2mesh import tsdf_reconstruction_from_saved_data, find_mesh_bottom_surface
import torch.nn.functional as F
import imageio
import numpy as np
from gsplat.rendering import rasterization_2dgs
from typing import List
from toolkit.camera import Camera

from toolkit.optical_flow import process_image_sequence
import sys
import genesis as gs


def proj_uv(xyz, cam):
    device = xyz.device
    intr = torch.eye(3).float().to(device)
    intr[0, 0] = cam.fx
    intr[1, 1] = cam.fy
    intr[0, 2] = cam.cx
    intr[1, 2] = cam.cy
    w2c = torch.eye(4).float().to(device)
    w2c[:3, :3] = torch.tensor(cam.R.T).to(device)
    w2c[:3, 3] = torch.tensor(cam.T).to(device)

    c_xyz = (w2c[:3, :3] @ xyz.T).T + w2c[:3, 3]
    i_xyz = (intr @ c_xyz.mT).mT  # (N, 3)
    uv = i_xyz[:, :2] / i_xyz[:, -1:].clip(1e-3) # (N, 2)
    return uv



class Scene_3d:
    def __init__(self, 
                 work_dir: str,
                 bottom_threshold: float = -1.0,
                 round_num = 1,
                 config = None
                 ):
        # read the gaussian
        self.work_dir = work_dir
        self.foreground_path = os.path.join(work_dir, "stage1_reconstruction/foreground/foreground.pt")
        self.background_path = os.path.join(work_dir, "stage1_reconstruction/background/background.pt")
        self.merged_gs = os.path.join(work_dir, "stage1_reconstruction/final_ckpt.pt")
        self.mesh_dir = os.path.join(work_dir, "stage2_forwardpass/mesh_output")
        self.simulation_dir = os.path.join(work_dir, "stage2_forwardpass/simulation")
        self.gaussians_dir = os.path.join(work_dir, "stage2_forwardpass/gaussians")
        self.render_dir = os.path.join(work_dir, "stage2_forwardpass/render_output")
        self.scene_info_path = os.path.join(work_dir, "stage2_forwardpass/scene_info.json")  # Add scene info path
        self.mesh_path = os.path.join(self.mesh_dir, "mesh_postprocessed.obj")
        self.config = config
        self.round_num = round_num
        if round_num == 1:
            self.physical_params_dir = os.path.join(work_dir, "stage2_forwardpass/physical_params", "round_1")
        else:
            self.physical_params_dir = os.path.join(work_dir, "stage2_forwardpass/physical_params", f"round_{round_num}")
        self.bottom_threshold = bottom_threshold
        # ensure the directory exists
        os.makedirs(self.mesh_dir, exist_ok=True)
        os.makedirs(self.simulation_dir, exist_ok=True)
        os.makedirs(self.gaussians_dir, exist_ok=True)
        os.makedirs(self.physical_params_dir, exist_ok=True)
        # 0. delete the old render_dir
        # import shutil
        # if os.path.exists(self.render_dir):
        #     shutil.rmtree(self.render_dir)
        os.makedirs(self.render_dir, exist_ok=True)
        # 2. generate the mesh
        self.setup_scene()
        self.camera_list = []
        self.depth_list = []

    def setup_scene(self):
        # 1. load the foreground gs and build the mesh
        foreground_gs = None
        foreground_gs = torch.load(self.foreground_path)
        foreground_gs = foreground_gs['splats']
        mesh, _ = tsdf_reconstruction_from_saved_data(
            work_dir=os.path.join(self.work_dir, "stage1_reconstruction"),
            output_dir=self.mesh_dir,
            bottom_threshold=self.bottom_threshold,
            voxel_length=self.config['simulator_config']['mesh_config']['voxel_length'],
            sdf_trunc_factor=self.config['simulator_config']['mesh_config']['sdf_trunc_factor'],
            depth_trunc=self.config['simulator_config']['mesh_config']['depth_trunc'],
            fill_holes=self.config['simulator_config']['mesh_config']['fill_holes'],
        )
        self.mesh = mesh

        # 2. get the mapping
        if foreground_gs is None:
            foreground_gs = torch.load(self.foreground_path)
            foreground_gs = foreground_gs['splats']
        # mapping = get_gaussian_mesh_mapping(foreground_gs, mesh, output_dir=self.mesh_dir)

        # 3. read the gs data
        gs_data = torch.load(self.foreground_path)
        gs_data = gs_data['splats']
        self.device = "cuda"
        self.gs_data = gs_data
        self.background_gs = torch.load(self.background_path)['splats']
        for key in gs_data:
            if isinstance(gs_data[key], torch.Tensor):
                gs_data[key] = gs_data[key].to(self.device)
        for key in self.background_gs:
            if isinstance(self.background_gs[key], torch.Tensor):
                self.background_gs[key] = self.background_gs[key].to(self.device)
        # self.mesh_mapping = mapping
        self.merged_gs = None
        
    def init_mesh_pos(self, mesh_pos):
        # Ensure mesh_pos is a torch tensor and on the correct device
        if isinstance(mesh_pos, np.ndarray):
            self.mesh_pos = torch.from_numpy(mesh_pos).to(self.device)
        else:
            self.mesh_pos = mesh_pos.clone().to(self.device)

        
    def render_gs(self, cam, sid):
        """Render 3D Gaussian splats"""
        # Ensure inputs are float32 type
        worldtocam = torch.linalg.inv(cam.camtoworld).float()
        K = cam.K.float()
        # Ensure on the correct device
        worldtocam = worldtocam.to(self.device)
        K = cam.K.to(self.device)

        # unsqueeze the first dimension
        worldtocam = worldtocam.unsqueeze(0)
        K = K.unsqueeze(0)

        # 1. Get Gaussian splat data and ensure all tensors are on the correct device
        means = self.merged_gs['means'].float().to(self.device)  # [N, 3] 
        quats = self.merged_gs['quats'].float().to(self.device)  # [N, 4]
        scales = torch.exp(self.merged_gs['scales'].float()).to(self.device)  # [N, 3]
        opacities = torch.sigmoid(self.merged_gs['opacities'].float()).to(self.device)  # [N]
        
        # 2. Get color data and normalize to [0,1] range
        sh0 = self.merged_gs['sh0'].float().to(self.device)
        shN = self.merged_gs['shN'].float().to(self.device)
        colors = torch.cat([sh0, shN], 1)  # [N, K, 3]
    
        # 4. Ensure all tensors are on the same device
        assert means.device == quats.device == scales.device == opacities.device == colors.device == worldtocam.device == K.device, f"Device mismatch: means={means.device}, quats={quats.device}, scales={scales.device}, opacities={opacities.device}, colors={colors.device}, worldtocam={worldtocam.device}, K={K.device}"
        
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
            width=cam.width,
            height=cam.height,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            sh_degree = 0,
            render_mode="RGB+ED",
            near_plane=0.2,
            far_plane=200.0,
            radius_clip=0.0,
        )

        def process_image(img):
            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
            if len(img.shape) == 2:
                img = img[..., None]
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            return np.clip(img, 0, 1)
        rgb = process_image(render_colors.squeeze(0))
        imageio.imwrite(
            os.path.join(self.render_dir, f"camera_{cam.img_id}/{sid}_rgb.png"),
            (rgb * 255).astype(np.uint8)
        )


    def load_camera_params(self, img_ids: List[int]):
        """
        Load camera parameters, supports camera ID matching and scene scale correction
        
        Args:
            img_ids: List of image IDs
            width: Image width
            height: Image height
        """
        load_dir = os.path.join(self.work_dir, "stage1_reconstruction", "camera_pose")
        camera_list = []
        for img_id in img_ids:
            load_path = os.path.join(load_dir, f"camera_{img_id:04d}.npz")  
            cam = Camera(img_id, load_path)
            camera_list.append(cam)
            # os.makedirs(os.path.join(self.render_dir, f"camera_{img_id}"), exist_ok=True)
            # os.makedirs(os.path.join(self.render_dir, f"camera_{img_id}_flow"), exist_ok=True)
        
        self.camera_list = camera_list

    def imgs2video(self, fps=30):
        for cam in self.camera_list:
            # Get all png files in the current directory
            png_files = [f for f in os.listdir(os.path.join(self.render_dir, f"camera_{cam.img_id}")) if f.endswith('rgb.png')]
            # Sort by frame number
            png_files.sort(key=lambda x: int(x.split('_')[0]))
            # Create video writer
            video_path = os.path.join(self.render_dir, f"camera_{cam.img_id}_render.mp4")
            with imageio.get_writer(video_path, fps=fps) as writer:
                for png_file in png_files:
                    img_path = os.path.join(os.path.join(self.render_dir, f"camera_{cam.img_id}"), png_file)
                    img = imageio.imread(img_path)
                    writer.append_data(img)
            print(f"Saved video to {video_path}")
        # self._save_4d_gs()
        # print("Saved 4d gs")
            process_image_sequence(os.path.join(self.render_dir, f"camera_{cam.img_id}"), os.path.join(self.render_dir, f"camera_{cam.img_id}_flow"))

    def _save_frame_gaussian(self, sid):
        # save as the pt
        torch.save({'splats': self.gs_data}, os.path.join(self.simulation_dir, f"frame_gs_{sid:04d}.pt"))

    def get_camera_by_id(self, cam_id):
        return self.camera_list[cam_id]

    def update(self, sid, jam, n):
        os.makedirs(os.path.join(self.physical_params_dir, f"entity_{n}"), exist_ok=True)

        # Get particle positions and ensure correct dimensions
        if isinstance(jam.material, gs.materials.Rigid):
            verts_local = np.array(jam.vgeoms[0].get_trimesh().vertices)  # [Nv, 3]
            
            # Get rigid body world coordinate position and rotation
            pos = jam.get_pos().cpu().numpy()  # [3]
            quat = jam.get_quat().cpu().numpy()  # [4] (w, x, y, z)
            
            # Build rotation matrix (from quaternion)
            w, x, y, z = quat
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ])
            
            # Apply transformation: verts_world = R @ verts_local.T + pos
            verts_world = (R @ verts_local.T).T + pos  # [Nv, 3]
            vel_world = jam.get_vel()
            vel_world = np.array(vel_world.cpu())
            current_mesh_pos = verts_world
            N = verts_world.shape[0]
            current_mesh_vel = vel_world[None,...].repeat(N, axis=0)
            current_mesh = np.concatenate([current_mesh_pos[None,...], current_mesh_vel[None,...]], axis=0)
            np.save(os.path.join(self.physical_params_dir, f"entity_{n}", f"{sid:04d}.npy"), current_mesh)
            if sid == 384:
                qpos = jam.get_qpos()
                qpos = np.array(qpos.cpu())
                vel = jam.get_dofs_velocity()
                vel = np.array(vel.cpu())
                result = np.concatenate([qpos, vel], axis=0)
                np.save(os.path.join(self.physical_params_dir, f"entity_{n}", f"qpos.npy"), result)


        elif isinstance(jam.material, gs.materials.PBD.Cloth):
            particle_start = jam.particle_start
            particle_end = jam.particle_end
            current_mesh_pos = jam.solver.particles.pos.to_numpy()[particle_start:particle_end]
            np.save(os.path.join(self.physical_params_dir, f"entity_{n}", f"{sid:04d}.npy"), current_mesh_pos)
        else:
            particle_start = jam.particle_start
            particle_end = jam.particle_end
            particles_xyzs = jam.solver.particles.pos.to_numpy()[particle_start:particle_end]
            particles_vels = jam.solver.particles.vel.to_numpy()[particle_start:particle_end]
            if len(particles_xyzs.shape) == 3:
                particles_xyzs = particles_xyzs[0]
                particles_vels = particles_vels[0]
            particles_xyzs = particles_xyzs[None, ...]
            particles_vels = particles_vels[None, ...]
            particles = np.concatenate([particles_xyzs, particles_vels], axis=0)
            np.save(os.path.join(self.physical_params_dir, f"entity_{n}", f"{sid:04d}.npy"), particles)

    def interpolate_camera_params(self, inter_num=9):
        """Interpolate camera parameters"""
        # from gsplat.examples.datasets.traj import generate_interpolated_path
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), "../gsplat/examples/datasets"))
        from traj import generate_interpolated_path
        # get the current camtoworld
        camtoworlds = []
        Ks = []
        for cam in self.camera_list:
            # Only take the first 3 rows, because generate_interpolated_path requires (n, 3, 4) format
            camtoworld_3x4 = cam.camtoworld[:3, :]
            camtoworlds.append(camtoworld_3x4)
            Ks.append(cam.K)
        
        # Convert list to numpy array to ensure correct shape
        camtoworlds = np.stack(camtoworlds, axis=0)  # Convert to (n, 3, 4) format
        
        interpolated_camtoworlds = generate_interpolated_path(camtoworlds, inter_num)
        cams_to_return = []
        for i, (interpolated_camtoworld) in enumerate(interpolated_camtoworlds):
            # Convert 3x4 matrix back to 4x4 format
            full_camtoworld = np.eye(4)
            full_camtoworld[:3, :] = interpolated_camtoworld
            
            # Create new camera object
            cam = Camera(img_id=i, camtoworld=full_camtoworld, K=Ks[0])  # Create a temporary camera object
            cams_to_return.append(cam)
            
        return cams_to_return


if __name__ == "__main__":
    import torch
    work_dir = "3d_result/jam"
    scene = Scene_3d(work_dir)
    
    # Use the fixed load_camera_params method to load camera parameters
    # Now automatically applies scene scale correction and displays camera ID information
    K, camtoworld = scene.load_camera_params(1)
    
    # Convert to torch tensor
    camtoworld = torch.from_numpy(camtoworld).float().cuda()
    K = torch.from_numpy(K).float().cuda()

    # Set rendering resolution
    width = 576
    height = 576

    # Add batch dimension
    camtoworld = camtoworld.unsqueeze(0)  # [1, 4, 4]
    K = K.unsqueeze(0)  # [1, 3, 3]

    results = scene.render_combined_scene(
        worldtocam=torch.linalg.inv(camtoworld),
        K=K,
        width=width,
        height=height
    )

    scene.save_renders(results, "view_specific")

    print(f"Saved view_specific_rgb.png")