conda activate cosmos-predict1
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python PerpetualWonder/reconstruction/gen3c_single_image.py --config_path examples/configs/play_doh.yaml 

conda activate pw

# Reconstruct the scene
python PerpetualWonder/reconstruction/colmap.py --input_dir1 3d_result/play_doh/stage1_reconstruction/play_doh_spin_left_images --input_dir2 3d_result/play_doh/stage1_reconstruction/play_doh_spin_right_images --output_dir 3d_result/play_doh
python PerpetualWonder/reconstruction/seg_video.py --config_path examples/configs/play_doh.yaml
python PerpetualWonder/reconstruction/simple_trainer_2dgs_seg.py --config examples/configs/play_doh.yaml
python PerpetualWonder/reconstruction/segment_gaussians.py --config_path examples/configs/play_doh.yaml

# Simulate the scene
python PerpetualWonder/forwardpass/simulation.py --config examples/configs/play_doh.yaml --round_num 1
python PerpetualWonder/reconstruction/simple_trainer_3dgs_dev.py --config examples/configs/play_doh.yaml 
python PerpetualWonder/forwardpass/render_particle_dynamics.py --config examples/configs/play_doh.yaml --round_num 1
python PerpetualWonder/forwardpass/simulation.py --config examples/configs/play_doh.yaml --round_num 2
python PerpetualWonder/forwardpass/render_particle_dynamics.py --config examples/configs/play_doh.yaml --round_num 2
python PerpetualWonder/forwardpass/simulation.py --config examples/configs/play_doh.yaml --round_num 3
python PerpetualWonder/forwardpass/render_particle_dynamics.py --config examples/configs/play_doh.yaml --round_num 3

# Optimize the scene
python PerpetualWonder/optimization/run_video_model.py --config examples/configs/play_doh.yaml
python PerpetualWonder/optimization/run_optim_4d.py --config examples/configs/play_doh.yaml --round_num 1 --semi_round True
python PerpetualWonder/optimization/run_optim_4d.py --config examples/configs/play_doh.yaml --round_num 2 --semi_round True
python PerpetualWonder/optimization/run_optim_4d.py --config examples/configs/play_doh.yaml --round_num 3 --semi_round True