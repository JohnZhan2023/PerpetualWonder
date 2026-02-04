![Title](assets/title.png)


# Long-Horizon Action-Conditioned 4D Scene Generation


<div align="center">

[![a](https://img.shields.io/badge/Website-PerpetualWonder-blue)](https://johnzhan2023.github.io/PerpetualWonder/)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2406.09394-red)](https://arxiv.org/abs/2406.09394)
[![twitter](https://img.shields.io/twitter/url?label=Koven_Yu&url=https%3A%2F%2Ftwitter.com%2FKoven_Yu)](https://x.com/Koven_Yu/status/1835769026934673595) -->
</div>


## Abstract

We introduce PerpetualWonder, a hybrid generative simulator that enables long-horizon, action-conditioned 4D scene generation from a single image. Current works fail at this task because their physical state is decoupled from their visual representation, which prevents generative refinements to update the underlying physics for subsequent interactions. PerpetualWonder solves this by introducing the first true closed-loop system. It features a novel unified representation that creates a bidirectional link between the physical state and visual primitives, allowing generative refinements to correct both the dynamics and appearance. It also introduces a robust update mechanism that gathers supervision from multiple viewpoints to resolve optimization ambiguity. Experiments demonstrate that from a single image, PerpetualWonder can successfully simulate complex, multi-step interactions from long-horizon actions, maintaining physical plausibility and visual consistency.


## Getting Started

For the installation to be done correctly, please proceed only with CUDA-compatible GPU available. It requires 80GB GPU memory to run.

**Tested Environment:**
- PyTorch: 2.7.1+cu126
- CUDA: 12.6

### Installation

This project requires two conda environments:
- **pw**: Main environment for running the primary functionality of the project
- **cosmos-predict1**: Used for generating multi-view videos with GEN3C

#### 1. Install Main Environment (pw)

Create and activate the main environment:

```bash
conda create -n pw python=3.10
conda activate pw
```

Install COLMAP:

```bash
conda install -c conda-forge colmap==3.11.1
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Segment Anything 2:

```bash
cd submodules/sam2
pip install -e .
cd ../..
```

Install submodules:

```bash
cd submodules/depth_diff_gaussian_rasterization_min
pip install -e . --no-build-isolation
cd ../diff-gaussian-rasterization-main
pip install -e . --no-build-isolation
cd ../..
```

Install Genesis:

```bash
cd Genesis
pip install -e .
cd ..
```

Install gsplat (for Gaussian rendering):

```bash
cd gsplat
pip install -e . --no-build-isolation
cd ..
```

Install simpe_knn:
```bash
git submodule update --init --recursive submodules/simple_knn
cd submodules/simple_knn
pip install -e . --no-build-isolation
cd ../..
```

#### 2. Install GEN3C Environment (cosmos-predict1)

Initialize the GEN3C submodule and set up the cosmos-predict1 environment:

```bash
git submodule update --remote --init submodules/GEN3C
cd submodules/GEN3C
conda env create --file cosmos-predict1.yaml
conda activate cosmos-predict1
pip install -r requirements.txt
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
pip install transformer-engine[pytorch]==1.12.0
git clone https://github.com/NVIDIA/apex
CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex
pip install git+https://github.com/microsoft/MoGe.git
cd ../..
```

### Run examples

We provide three example cases: **play_doh**, **dumpling** and **jam**. Each example has a corresponding shell script in the `scripts/` directory that contains the complete pipeline commands.

The pipeline consists of three main stages:

#### Stage 1: Scene Reconstruction

First, generate multi-view videos using GEN3C. Note that you need to modify `examples/configs/{scene_name}.yaml` to set `gen3c_config.trajectory` to `["spin_right", "spin_left"]`:

```bash
conda activate cosmos-predict1
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python PerpetualWonder/reconstruction/gen3c_single_image.py --config_path examples/configs/jam.yaml
```

Then, switch to the main environment and perform scene reconstruction:

```bash
conda activate pw
python PerpetualWonder/reconstruction/colmap.py 3d_result/jam/images 3d_result/jam
python PerpetualWonder/reconstruction/seg_video.py --config_path examples/configs/jam.yaml
python PerpetualWonder/reconstruction/simple_trainer_2dgs_seg.py --config examples/configs/jam.yaml
python PerpetualWonder/reconstruction/segment_gaussians.py --config_path examples/configs/jam.yaml
```

#### Stage 2: Forward Pass and Backward Optimization

After completing the scene reconstruction, the pipeline alternates between forward pass and backward optimization for multiple rounds (Here, we take the first round for example.).

**Forward Pass:** Run a simulator simulation, then use the simulated physical particle trajectories to move the Gaussians and render the results:

```bash
python PerpetualWonder/forwardpass/simulation.py --config examples/configs/jam.yaml --round_num 1
python PerpetualWonder/forwardpass/render_particle_dynamics.py --config examples/configs/jam.yaml --round_num 1
```

**Backward Optimization:** Use the rendered results to refine with the video model. 

For the first refinement, modify the corresponding `config.yaml` file by setting `simulator_config.camera_list` to `[121]`, which refines the front view, then optimize the Gaussian scene:

```bash
python PerpetualWonder/optimization/run_video_model_v2.py --config examples/configs/jam.yaml --round_num 1
python PerpetualWonder/optimization/run_optim_4d_v2.py --config examples/configs/jam.yaml --round_num 1 --semi_round False
```

Then perform side-view refinement by setting `simulator_config.camera_list` to `[83, 159]`:

```bash
python PerpetualWonder/optimization/run_video_model_v2.py --config examples/configs/jam.yaml --round_num 1
python PerpetualWonder/optimization/run_optim_4d_v2.py --config examples/configs/jam.yaml --round_num 1 --semi_round True
```

For subsequent rounds (round 2 and round 3), simply change the `round_num` parameter and repeat the forward pass and backward optimization steps.

#### Notes

**Conda Environments:** Two conda environments are required:
- `cosmos-predict1`: Used only for GEN3C to generate multi-view videos (activate with `conda activate cosmos-predict1`)
- `pw`: Used for all subsequent steps in the pipeline (activate with `conda activate pw`)

**Reproducibility:** : Due to the stochastic nature of video model generation, to help reproduce the results in the paper, we will provide some intermediate results in `3d_result/`:
- `3d_result/{scene_name}/stage1_reconstruction/3d/images`: Generated by GEN3C (multi-view video generation)
- `3d_result/{scene_name}/stage3_optimization/go_with_flow`: Results from video model refinement


### How to add more examples?

To add a new example, follow these steps:

1. **Create image directory**: Add a new folder named `{scene_name}` under `examples/imgs/` and place images with dimensions 1280×704 in that folder.

2. **Create configuration file**: Create a `{scene_name}.yaml` configuration file in `examples/configs/` directory to configure your scene.

3. **Configure simulation**: Add simulation-specific configuration in `PerpetualWonder/forwardpass/simulation_config.py` based on the type of physics simulation you want to perform.

## Acknowledgement

Our code references and builds upon the following open-source projects:

- [GEN3C](https://github.com/nv-tlabs/GEN3C)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- [SAM2](https://github.com/facebookresearch/sam2)
- [HUGS](https://github.com/hyzhou404/HUGS)
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)
- [simple_knn](https://github.com/shuyueW1991/simple_knn_illustration)

We are grateful to the authors and contributors of these projects for their valuable work.

