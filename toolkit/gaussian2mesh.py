import torch
import numpy as np
import trimesh
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import os
from scipy.spatial import ConvexHull
import imageio
import cv2
import json
# from oriented_pbd.diff_simulator_v3 import PRESET_Z_VALUE, gs_to_pt3d, pt3d_to_gs
PRESET_Z_VALUE = 0.0 # 10.0

def get_gaussian_mesh_mapping(gaussian_data, mesh, output_dir="output"):
    """
    Get the mapping relationship between Gaussian points and mesh vertices.
    This version fixes coordinate system issues to prevent mapping inversion caused by Y-axis flipping.

    Matching process:
    1. Coordinate system transformation: Assume the mesh comes from the standard world coordinate system (Y-up),
       and transform it to the pt3d coordinate system where Gaussian points are located.
       - Only flip the X-axis to unify left-handed and right-handed coordinate systems.
    2. Centroid alignment: Move the transformed mesh to the same center position as the Gaussian point cloud.
    3. Nearest neighbor matching: Find the nearest mesh vertex for each Gaussian point.
    
    Args:
        gaussian_data (dict): Dictionary containing Gaussian point data, expected to have 'means' key.
        mesh (trimesh.Trimesh): trimesh object.
        output_dir (str): Output directory path.
    
    Returns:
        dict: Mapping dictionary containing 'gaussian_to_mesh' and 'mesh_to_gaussians'.
    """
    os.makedirs(output_dir, exist_ok=True)
    gaussian_vertices = gaussian_data['means'].detach().cpu().numpy()
    mesh_vertices = np.array(mesh.vertices)
    
    print(f"Starting matching: {len(gaussian_vertices)} Gaussian points -> {len(mesh_vertices)} mesh vertices")


    mesh_vertices_transformed = mesh_vertices.copy()
    # visualize the points and gaussian_vertices using visualize_3d_points
    sampled_gaussian_vertices = gaussian_vertices[::10]
    visualize_3d_points(
        [mesh_vertices_transformed, sampled_gaussian_vertices], title="Mesh Vertices", save_path=os.path.join(output_dir, "mesh_vertices_and_gaussian_vertices.png")
    )

    # 4. Nearest neighbor matching
    print("Performing nearest neighbor matching...")
    # Use ball_tree/kd_tree to accelerate search for large point clouds
    mesh_tree = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(mesh_vertices_transformed)
    distances, mesh_indices = mesh_tree.kneighbors(gaussian_vertices)
    gaussian_to_mesh = mesh_indices.flatten()
    
    # 5. Build reverse mapping from mesh -> gaussians
    mesh_to_gaussians = [[] for _ in range(len(mesh_vertices_transformed))]
    for gaussian_idx, mesh_idx in enumerate(gaussian_to_mesh):
        mesh_to_gaussians[mesh_idx].append(gaussian_idx)
    
    # 6. Report and save
    avg_distance = np.mean(distances)
    matched_vertices_count = sum(1 for g_list in mesh_to_gaussians if g_list)
    print(f"Matching completed! Average distance: {avg_distance:.4f}.")
    print(f"{matched_vertices_count}/{len(mesh_vertices)} mesh vertices are matched by at least one Gaussian point.")
    
    mapping = {
        'gaussian_to_mesh': gaussian_to_mesh,
        'mesh_to_gaussians': mesh_to_gaussians
    }
    
    mapping_file = os.path.join(output_dir, "gaussian_mesh_mapping.pkl")
    import pickle
    with open(mapping_file, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"Mapping relationship saved to: {mapping_file}")
    
    return mapping


def tsdf_reconstruction_from_saved_data(work_dir, output_dir="output", 
                                       voxel_length=0.02, sdf_trunc_factor=2.0,  # Increase truncation factor to expand mesh
                                       depth_trunc=20.0,
                                       fill_holes=True, bottom_threshold=0.0):
    """
    Perform TSDF reconstruction from saved RGB-D images and camera poses
    
    Args:
        work_dir: Working directory containing the following structure:
            - work_dir/rgbd/rgb/ - RGB image folder
            - work_dir/rgbd/depth/ - Depth image folder  
            - work_dir/camera_pose/ - Camera pose folder
        output_dir: Output directory path (default "output")
        voxel_length: TSDF voxel size (default 0.02)
        sdf_trunc_factor: SDF truncation factor relative to voxel_length (default 1.5, lower to capture more translucent regions)
        depth_trunc: Depth truncation value (default 10.0)
        fill_holes: Whether to fill holes (default True)
        bottom_threshold: Bottom threshold, parts greater than this value will be cut off (optional)
    
    Returns:
        trimesh.Trimesh: Generated mesh object
    """
    import glob
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build paths
    rgb_dir = os.path.join(work_dir, "rgbd", "masked_rgb")
    depth_dir = os.path.join(work_dir, "rgbd", "depth")
    camera_pose_dir = os.path.join(work_dir, "camera_pose")
    
    # 1. load the rgb images
    # rgb_dir = os.path.join(work_dir, "3d", "images")
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    mask_dir = os.path.join(work_dir, "3d", "masks")
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    rgb_images = []
    for i, rgb_file in enumerate(rgb_files):
        rgb = imageio.imread(rgb_file)
        rgb_images.append(rgb)

    # Load depth images
    depth_images = []
    for i in range(len(rgb_files)):
        depth_file = os.path.join(depth_dir, f"depth_{i:04d}.npy")
        if not os.path.exists(depth_file):
            raise ValueError(f"Depth file does not exist: {depth_file}")
        depth = np.load(depth_file)
        depth_images.append(depth)
        
    camera_folder = os.path.join(work_dir, "camera_pose")
    num_views = len(os.listdir(camera_folder))
    poses = []
    intrinsics = []
    for i in range(num_views):
        pose_file = os.path.join(camera_folder, f"camera_{i:04d}.npz")
        pose_data = np.load(pose_file)
        poses.append(pose_data['camtoworld'])
        intrinsics.append(pose_data['K'])

    # Use the new tsdf_fusion_from_rgbd function
    mesh_o3d = tsdf_fusion_from_rgbd(
        rgb_images=rgb_images,
        depth_images=depth_images,
        poses=poses,
        intrinsics=intrinsics,
        voxel_size=voxel_length,
        sdf_trunc=sdf_trunc_factor * voxel_length,
        depth_trunc=depth_trunc,
        output_dir=output_dir,
        bottom_threshold=bottom_threshold,
        fill_holes=fill_holes
    )
    
    # Convert to trimesh format
    mesh_vertices = np.asarray(mesh_o3d.vertices)
    mesh_faces = np.asarray(mesh_o3d.triangles)
    mesh_colors = np.asarray(mesh_o3d.vertex_colors)
    
    print(f"Reconstruction completed: {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")
    
    # Create trimesh object
    mesh = trimesh.Trimesh(
        vertices=mesh_vertices,
        faces=mesh_faces,
        vertex_colors=mesh_colors,
        process=False
    )
    
    # Mesh post-processing: remove non-main components, ground completion
    mesh = postprocess_mesh_connect_and_ground(mesh)
    # Save processed mesh
    mesh_path = os.path.join(output_dir, "mesh_postprocessed.obj")
    mesh.export(mesh_path)
    print(f"Post-processed mesh saved to: {mesh_path}")
    
    return mesh, mesh

def find_mesh_bottom_surface(mesh, percentile=10, min_support_ratio=0.001):
    """
    Intelligently detect the bottom surface position of the mesh
    
    Args:
        mesh: trimesh.Trimesh object
        percentile: Percentile used to determine bottom surface height (default 5, i.e., lowest 5% of points)
        min_support_ratio: Minimum support ratio to ensure sufficient support points on the bottom surface (default 0.3)
    
    Returns:
        float: z-coordinate of the bottom surface
        dict: Dictionary containing bottom surface analysis information
    """
    vertices = mesh.vertices
    faces = mesh.faces
    
    # 1. Get z-coordinates of all vertices
    z_coords = vertices[:, 2]
    
    # 2. Calculate statistics of z-coordinates
    z_min = z_coords.min()
    z_max = z_coords.max()
    z_mean = z_coords.mean()
    z_std = z_coords.std()
    
    # 3. Use percentile method to find bottom surface candidate region
    z_threshold = np.percentile(z_coords, percentile)
    
    # 4. Find vertices below the threshold
    bottom_vertices_mask = z_coords <= z_threshold
    bottom_vertices = vertices[bottom_vertices_mask]
    
    # 5. Calculate bottom surface support area
    # Project bottom vertices onto xy plane
    bottom_xy = bottom_vertices[:, :2]
    
    # Calculate convex hull area of bottom surface in xy plane
    if len(bottom_xy) >= 3:
        try:
            hull = ConvexHull(bottom_xy)
            bottom_area = hull.volume  # In 2D, volume is actually area
        except:
            # If convex hull calculation fails, use simple bounding box area
            bottom_area = (bottom_xy[:, 0].max() - bottom_xy[:, 0].min()) * \
                         (bottom_xy[:, 1].max() - bottom_xy[:, 1].min())
    else:
        bottom_area = 0
    
    # 6. Calculate projection area of entire mesh in xy plane
    all_xy = vertices[:, :2]
    if len(all_xy) >= 3:
        try:
            hull_all = ConvexHull(all_xy)
            total_area = hull_all.volume
        except:
            total_area = (all_xy[:, 0].max() - all_xy[:, 0].min()) * \
                        (all_xy[:, 1].max() - all_xy[:, 1].min())
    else:
        total_area = 1
    
    # 7. Calculate support ratio
    support_ratio = bottom_area / total_area if total_area > 0 else 0
    
    # 8. If support ratio is too small, adjust threshold
    if support_ratio < min_support_ratio:
        # Gradually increase percentile until sufficient support is found
        for p in range(percentile + 5, 50, 5):
            z_threshold = np.percentile(z_coords, p)
            bottom_vertices_mask = z_coords <= z_threshold
            bottom_vertices = vertices[bottom_vertices_mask]
            bottom_xy = bottom_vertices[:, :2]
            
            if len(bottom_xy) >= 3:
                try:
                    hull = ConvexHull(bottom_xy)
                    bottom_area = hull.volume
                except:
                    bottom_area = (bottom_xy[:, 0].max() - bottom_xy[:, 0].min()) * \
                                 (bottom_xy[:, 1].max() - bottom_xy[:, 1].min())
            else:
                bottom_area = 0
            
            support_ratio = bottom_area / total_area if total_area > 0 else 0
            if support_ratio >= min_support_ratio:
                break
    
    # 9. Use density analysis to find the most stable bottom surface height
    # Perform density analysis in the bottom surface region
    bottom_vertices_mask = z_coords <= z_threshold
    bottom_z_coords = z_coords[bottom_vertices_mask]
    
    if len(bottom_z_coords) > 0:
        # Use histogram to find z-value with highest density
        hist, bin_edges = np.histogram(bottom_z_coords, bins=20)
        max_density_idx = np.argmax(hist)
        stable_bottom_z = (bin_edges[max_density_idx] + bin_edges[max_density_idx + 1]) / 2
    else:
        stable_bottom_z = z_threshold
    
    # 10. Return analysis results
    analysis_info = {
        'z_min': z_min,
        'z_max': z_max,
        'z_mean': z_mean,
        'z_std': z_std,
        'z_threshold': z_threshold,
        'stable_bottom_z': stable_bottom_z,
        'support_ratio': support_ratio,
        'bottom_area': bottom_area,
        'total_area': total_area,
        'bottom_vertex_count': len(bottom_vertices),
        'total_vertex_count': len(vertices),
        'percentile_used': percentile
    }
    
    return stable_bottom_z, analysis_info


def visualize_3d_points(points_list, labels=None, colors=None, point_sizes=None, title="3D Points Visualization", 
                       axis_length=1.0, save_path=None, show_plot=True):
    """
    Visualize multiple point clouds and coordinate systems in 3D space, rendering three images from different viewpoints
    
    Args:
        points_list: List of point clouds, each element is a numpy array with shape (N, 3)
        labels: List of labels for each point cloud, used for legend
        colors: List of colors for each point cloud
        point_sizes: List of point sizes for each point cloud
        title: Image title
        axis_length: Coordinate axis length
        save_path: Path to save the image, if None then don't save
        show_plot: Whether to display the image
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Set default values
    if not isinstance(points_list, list):
        points_list = [points_list]
    if labels is None:
        labels = [f"Points {i+1}" for i in range(len(points_list))]
    if colors is None:
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        colors = colors[:len(points_list)]
    if point_sizes is None:
        point_sizes = [1] * len(points_list)
    
    # Define three different viewpoints
    views = [
        {'elev': 20, 'azim': 45, 'name': 'perspective'},  # Perspective view
        {'elev': 90, 'azim': 0, 'name': 'top'},          # Top view
        {'elev': 0, 'azim': 0, 'name': 'front'}          # Front view
    ]
    
    # Create a large figure containing three subplots
    fig = plt.figure(figsize=(24, 8))
    
    # Calculate point cloud range
    all_points = np.concatenate(points_list, axis=0)
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = np.mean(all_points[:, 0])
    mid_y = np.mean(all_points[:, 1])
    mid_z = np.mean(all_points[:, 2])
    
    # Draw three plots from different viewpoints
    for i, view in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        
        # Draw each point cloud
        for points, label, color, size in zip(points_list, labels, colors, point_sizes):
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=color, s=size, label=label, alpha=0.6)
        
        # Draw coordinate system
        origin = np.zeros(3)
        # X-axis (red)
        ax.quiver(origin[0], origin[1], origin[2], 
                 axis_length, 0, 0, color='red', arrow_length_ratio=0.1)
        # Y-axis (green)
        ax.quiver(origin[0], origin[1], origin[2], 
                 0, axis_length, 0, color='green', arrow_length_ratio=0.1)
        # Z-axis (blue)
        ax.quiver(origin[0], origin[1], origin[2], 
                 0, 0, axis_length, color='blue', arrow_length_ratio=0.1)
        
        # Set viewpoint
        ax.view_init(elev=view['elev'], azim=view['azim'])
        
        # Set title
        ax.set_title(f"{title} - {view['name'].capitalize()} View")
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal axis scale
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add grid
        ax.grid(True)
        
        # Add legend
        ax.legend()
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save image
    if save_path is not None:
        # Modify save path, add viewpoint information
        base_path = save_path.rsplit('.', 1)[0]
        ext = save_path.rsplit('.', 1)[1]
        plt.savefig(f"{base_path}_multi_view.{ext}", dpi=300, bbox_inches='tight')
        print(f"Multi-view image saved to: {base_path}_multi_view.{ext}")
    
    # Display image
    if show_plot:
        plt.show()
    else:
        plt.close()


# ===================== Mesh post-processing: remove non-main components, ground completion =====================

# ===================== Modify postprocess_mesh_connect_and_ground to call strip surface coverage =====================
def postprocess_mesh_connect_and_ground(mesh):
    """
    Post-process the mesh:
    1. Keep the largest connected component (main body)
    2. Strip surface coverage in positive x direction
    Args:
        mesh: trimesh.Trimesh object
    Returns:
        trimesh.Trimesh: Processed mesh
    """
    import trimesh

    # 1. Keep the largest connected component
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh_combined = max(components, key=lambda m: len(m.vertices))
        print(f"Kept largest connected component: {len(mesh.vertices)} vertices")
    else:
        mesh_combined = mesh

    return mesh_combined


def tsdf_fusion_from_rgbd(
    rgb_images, depth_images, poses, intrinsics,
    voxel_size=0.02, sdf_trunc=0.01, depth_trunc=10.0,  # Adjust these parameters to capture more translucent regions
    output_dir="output", bottom_threshold=0.2, fill_holes=True
):
    """
    Reconstruct mesh from RGB-D sequence using TSDF voxel fusion, supports mask removal of background.
    Uses smaller sdf_trunc value to capture translucent regions of smoke.

    Args:
        rgb_images: List of RGB images, each element is a numpy array (H, W, 3), value range [0, 255] or [0, 1]
        depth_images: List of depth images, each element is a numpy array (H, W)
        poses: List of camera poses, each element is a 4x4 numpy array (camera to world)
        intrinsics: List of camera intrinsics, each element is a 3x3 or 4x4 numpy array
        voxel_size: TSDF voxel size (default 0.02)
        sdf_trunc: SDF truncation value, smaller values can capture more translucent regions (default 0.03)
        depth_trunc: Depth truncation value (default 10.0)
        output_dir: Output directory path
        bottom_threshold: Bottom threshold, parts greater than this value will be cut off (optional)
        fill_holes: Whether to fill holes (default True)
    """
    import open3d as o3d
    import numpy as np
    import os

    os.makedirs(output_dir, exist_ok=True)
    print("Starting TSDF voxel fusion ...")
    print(f"voxel_size: {voxel_size}, sdf_trunc: {sdf_trunc}, depth_trunc: {depth_trunc}")

    # Use smaller weight_threshold to capture translucent regions
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i, (rgb, depth, pose, intrinsic) in enumerate(zip(rgb_images, depth_images, poses, intrinsics)):
        # Process RGB
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
        # Process depth
        depth = depth.astype(np.float32)

        # Open3D image objects
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth)

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )

        # Process intrinsics
        if intrinsic.shape == (3, 3):
            intrinsic_4x4 = np.eye(4)
            intrinsic_4x4[:3, :3] = intrinsic
        else:
            intrinsic_4x4 = intrinsic
        fx, fy = intrinsic_4x4[0, 0], intrinsic_4x4[1, 1]
        cx, cy = intrinsic_4x4[0, 2], intrinsic_4x4[1, 2]
        h, w = rgb.shape[:2]
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
        )

        # Extrinsics
        extrinsic = np.linalg.inv(pose)

        # Fusion
        volume.integrate(rgbd, intrinsic_o3d, extrinsic)

    print("Extracting mesh ...")
    mesh_o3d = volume.extract_triangle_mesh()
    
    # Cut mesh according to bottom_threshold
    if bottom_threshold is not None:
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        colors = np.asarray(mesh_o3d.vertex_colors)
        
        # Find vertices with z-coordinate greater than or equal to threshold
        valid_vertex_mask = vertices[:, 2] >= bottom_threshold
        valid_vertex_indices = np.where(valid_vertex_mask)[0]
        
        # Create vertex index mapping
        vertex_map = np.full(len(vertices), -1, dtype=int)
        vertex_map[valid_vertex_indices] = np.arange(len(valid_vertex_indices))
        
        # Filter faces: only keep faces where all vertices are within threshold
        valid_faces = []
        for face in faces:
            if all(vertex_map[vertex_idx] != -1 for vertex_idx in face):
                # Remap vertex indices
                new_face = [vertex_map[vertex_idx] for vertex_idx in face]
                valid_faces.append(new_face)
        
        # Create new mesh
        new_vertices = vertices[valid_vertex_indices]
        new_faces = np.array(valid_faces)
        new_colors = colors[valid_vertex_indices] if len(colors) > 0 else np.array([])
        
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(new_vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(new_faces)
        if len(new_colors) > 0:
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(new_colors)
        
        print(f"After cutting: {len(new_vertices)} vertices, {len(new_faces)} faces")
    
    # Fill holes
    if fill_holes and len(mesh_o3d.vertices) > 0:
        print("Filling holes using trimesh...")
        # Convert to trimesh format
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        colors = np.asarray(mesh_o3d.vertex_colors)
        
        # Create trimesh object
        mesh_trimesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors if len(colors) > 0 else None,
            process=False
        )
        
        # Use trimesh to fill holes
        mesh_trimesh.fill_holes()
        # Convert back to Open3D format
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
        
        print(f"After filling holes: {len(mesh_o3d.vertices)} vertices, {len(mesh_o3d.triangles)} faces")

    return mesh_o3d

