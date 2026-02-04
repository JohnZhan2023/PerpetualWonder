
import genesis as gs
import taichi as ti
import numpy as np
import math

# Wind force function definitions - reference Genesis/test_clothes_two.py


@ti.func
def anti_gravity(pos, vel, t, i):
    # Create wind acceleration
    acceleration = ti.Vector.zero(gs.ti_float, 3)
    
    # Main horizontal wind force
    wind_x_strength = 9.8  # Constant wind strength

    wind_x = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
    force_x = wind_x * wind_x_strength * t / 0.21

    # Combine all wind forces
    acceleration = force_x 
    
    return acceleration


def config_scene_3d(config, scene_3d):
    if config["scene_name"] == "jam_2":
        sim_config = config['simulator_config']
        
        # Get first entity config from entities list
        entity_config = sim_config['entities'][0]
        # Hardcoded material parameters for high-viscosity liquid (simulating jam spreading)
        E = 1e5        # Small elastic modulus, only for numerical stability
        nu = 0.49      # Near incompressible
        rho = 1000.0   # Density
        viscous = True # Enable viscous term
        mu = 100000.0  # High viscosity, flows slowly and thickly

        # Load bounds from YAML and convert to tuple
        min_bound = tuple(float(x) for x in sim_config['min_bound'])
        max_bound = tuple(float(x) for x in sim_config['max_bound'])
        if scene_3d.round_num == 1:
            scene = gs.Scene(
                show_viewer = False,
                sim_options=gs.options.SimOptions(
                    dt = float(sim_config['dt']),
                    substeps = int(sim_config['substeps']),
                ),
                vis_options = gs.options.VisOptions(
                    show_world_frame = True,
                    world_frame_size = 1.0,
                    show_link_frame  = False,
                    show_cameras     = False,
                    plane_reflection = True,
                    ambient_light    = (0.1, 0.1, 0.1),
                    visualize_mpm_boundary = True,
                ),
                renderer=gs.renderers.Rasterizer(),
                mpm_options = gs.options.MPMOptions(
                    lower_bound = min_bound,
                    upper_bound = max_bound,
                    grid_density = int(sim_config['grid_density']),
                    particle_size = float(entity_config['particle_size']),
                    gravity = (0, 0, -9.8),
                )
            )
            force_field = gs.force_fields.Custom(anti_gravity)
            force_field.activate()
            scene.add_force_field(force_field = force_field)
        elif scene_3d.round_num == 2:
            scene = gs.Scene(
                show_viewer = False,
                sim_options=gs.options.SimOptions(
                    dt = float(sim_config['dt']),
                    substeps = int(sim_config['substeps']),
                ),
                vis_options = gs.options.VisOptions(
                    show_world_frame = True,
                    world_frame_size = 1.0,
                    show_link_frame  = False,
                    show_cameras     = False,
                    plane_reflection = True,
                    ambient_light    = (0.1, 0.1, 0.1),
                    visualize_mpm_boundary = True,
                ),
                renderer=gs.renderers.Rasterizer(),
                mpm_options = gs.options.MPMOptions(
                    lower_bound = min_bound,
                    upper_bound = max_bound,
                    grid_density = int(sim_config['grid_density']),
                    particle_size = float(entity_config['particle_size']),
                    gravity = (0, 0, 0),
                )
            )
        
            force_field = gs.force_fields.Wind(
                direction = (0, 1, 0),
                strength = 3.0,
                radius = 0.8,
                center = (0, 0, 1),
            )
            force_field.activate()
            scene.add_force_field(force_field = force_field)
        elif scene_3d.round_num == 3:
            scene = gs.Scene(
                show_viewer = False,
                sim_options=gs.options.SimOptions(
                    dt = float(sim_config['dt']),
                    substeps = int(sim_config['substeps']),
                ),
                vis_options = gs.options.VisOptions(
                    show_world_frame = True,
                    world_frame_size = 1.0,
                    show_link_frame  = False,
                    show_cameras     = False,
                    plane_reflection = True,
                    ambient_light    = (0.1, 0.1, 0.1),
                    visualize_mpm_boundary = True,
                ),
                renderer=gs.renderers.Rasterizer(),
                mpm_options = gs.options.MPMOptions(
                    lower_bound = min_bound,
                    upper_bound = max_bound,
                    grid_density = int(sim_config['grid_density']),
                    particle_size = float(entity_config['particle_size']),
                    gravity = (0, 0, 0),
                )
            )
        
            force_field = gs.force_fields.Wind(
                direction = (0, -1, 0),
                strength = 7.0,
                radius = 0.9,
                center = (0, 0, 1),
            )
            force_field.activate()
            scene.add_force_field(force_field = force_field)

        jam = scene.add_entity(
            material = gs.materials.MPM.Liquid(
                E = E,
                nu = nu,
                rho = rho,
                viscous = viscous,
                mu = mu,
                sampler = "pbs",
            ),
            morph = gs.morphs.Mesh(file=scene_3d.mesh_path, pos = (0, 0, -sim_config['lift_height']), euler = (0, 0, 0))
        )

        plane = scene.add_entity(gs.morphs.Plane(
            pos = (0, 0, 0),
            euler = (0, sim_config['tilt_angle'], 0)
        ))
        return scene
    elif config["scene_name"] == "dumpling":
        sim_config = config['simulator_config']
        min_bound = tuple(float(x) for x in sim_config['min_bound'])
        max_bound = tuple(float(x) for x in sim_config['max_bound'])
        if scene_3d.round_num == 1:
            scene = gs.Scene(
                show_viewer = False,
                sim_options=gs.options.SimOptions(
                    dt = float(sim_config['dt']),
                    substeps = int(sim_config['substeps']),
                    gravity = (0, 0, -19.8),
                ),
                mpm_options=gs.options.MPMOptions(
                    lower_bound = min_bound,
                    upper_bound = max_bound,
                    particle_size = float(sim_config['particle_size']),
                    grid_density = sim_config['grid_density'], 
                    gravity = (0, 0, 0),
                ),            
            )
        elif scene_3d.round_num == 2 or scene_3d.round_num == 3:
            scene = gs.Scene(
                show_viewer = False,
                sim_options=gs.options.SimOptions(
                    dt = float(sim_config['dt']),
                    substeps = int(sim_config['substeps']),
                    gravity = (0, 0, 0),
                ),
                mpm_options=gs.options.MPMOptions(
                    lower_bound = min_bound,
                    upper_bound = max_bound,
                    particle_size = float(sim_config['particle_size']),
                    grid_density = sim_config['grid_density'],
                    gravity = (0, 0, 0),
                ),            
            )



        dough = scene.add_entity(
            material = gs.materials.MPM.Elastic(
                E = 8e3,  
                nu = 0.45,  
                rho = 1000.0,  
                model = "corotation",
            ),
            morph = gs.morphs.Mesh(
                file = "3d_result/dumpling/dough.obj",
            )
        )
        
        rolling_pin = scene.add_entity(
            material = gs.materials.Rigid(),
            morph = gs.morphs.Mesh(
                file = "3d_result/dumpling/sparse_rolling_transformed.obj",
            )
        )
        plane = scene.add_entity(
            material = gs.materials.Rigid(),
            morph = gs.morphs.Plane(
                pos = (0, 0, 0.0),
                euler = (0, 0, 0),
            )
        )

        wind_force_center = (
            -1, 0, 0,
        )
        wind_force_radius = 1.0
        wind_force_direction = (
            0,
            0,
            -1
        )
        force_field = gs.force_fields.Wind(
            direction = wind_force_direction,
            strength = 10.0,
            radius = wind_force_radius,
            center = wind_force_center,
        )
        force_field.activate()
        scene.add_force_field(
            force_field = force_field
        )
        return scene
    elif config["scene_name"] == "play_doh":
        sim_config = config['simulator_config']
        lower_bound = tuple(float(x) for x in sim_config['min_bound'])
        upper_bound = tuple(float(x) for x in sim_config['max_bound'])
        
        # Create scene object
        scene = gs.Scene(
            show_viewer = False,
            sim_options=gs.options.SimOptions(
                dt       = config['simulator_config']['dt'],
                substeps = config['simulator_config']['substeps'],
                gravity = (0.0, 0.0, 9.8),
            ),
            pbd_options=gs.options.PBDOptions(
                lower_bound = lower_bound,
                upper_bound = upper_bound,
                particle_size = float(sim_config['particle_size']),
                gravity = (0.0, 0.0, 9.8),  
            ),
            mpm_options = gs.options.MPMOptions(
                lower_bound = lower_bound,
                upper_bound = upper_bound,
                grid_density = int(sim_config['grid_density']),
                particle_size = float(sim_config['particle_size']),
                gravity = (0, 0, 9.8),
            )
        )
        # Use the MPM Elastic material
        play_doh = scene.add_entity(
            material = gs.materials.MPM.Snow(),
            morph = gs.morphs.Mesh(file=scene_3d.mesh_path),
        )
        plane = scene.add_entity(
            morph = gs.morphs.Plane(
                pos = (0, 0, -1.1),
                euler = (0, 180, 0),
            ),
            material = gs.materials.Rigid(
                friction = 5.0,
                coup_friction = 1.5,
                coup_softness = 0.005,
                gravity_compensation = 1.0,
            ),
        )
        if scene_3d.round_num == 1:
            # Add wind
            force_field = gs.force_fields.Wind(
                direction = (0, 0, 1),
                strength = 20.0,
                radius = 15.0,
                center = (9.0, 1.0, -1.0),
            )
            force_field.activate()
            scene.add_force_field(force_field = force_field)
        elif scene_3d.round_num == 2:
            # Add wind
            force_field = gs.force_fields.Wind(
                direction = (0, 1, 1),
                strength = 20.0,
                radius = 4.0,
                center = (9.0, 4.0, -2.0),
            )
            force_field.activate()
            scene.add_force_field(force_field = force_field)
        elif scene_3d.round_num == 3:
            # Add wind
            force_field = gs.force_fields.Wind(
                direction = (0, -1, 0.5),
                strength =27.0,
                radius = 100.0,
                center = (9.0, 4.0, -2.0),
            )
            force_field.activate()
            scene.add_force_field(force_field = force_field)            
            


        return scene
