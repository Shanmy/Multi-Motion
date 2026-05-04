# Towards Open Domain Text-Driven Synthesis of Multi-Person Motions (ECCV 2024)

[[Paper]](https://shanmy.github.io/Multi-Motion/static/pdfs/Multi_Person_Human_Motion_Diffusion_Model.pdf) | [[Project Page]](https://shanmy.github.io/Multi-Motion/)

Each subfolder has a README.md explaining its purpose. A brief summary and setup instructions are below.

---

## Installation

```bash
# 1. Install shared packages
pip install -e .

# 2. Install diffusion-module extras
pip install -r diffusion/requirements.txt

# 3. Install CLIP (required for text conditioning)
pip install git+https://github.com/openai/CLIP.git

# 4. Install detectron2 (required for data collection only)
#    See https://detectron2.readthedocs.io/en/latest/tutorials/install.html

# 5. Obtain SMPL-A model files (required for visualization)
#    See smpl/README.md for instructions
```

---

## Training

The model uses a **two-stage** training pipeline. Stage 1 (pose model) must be trained before Stage 2 (motion model).

### Prerequisites

Before training, edit the config files in `diffusion/configs_train/` to set:
- `out_dir` — directory where checkpoints and logs will be saved
- Dataset paths inside `dataset=dict(...)` (see `pose_datasets/` loaders for the expected format)

To enable FID evaluation during training, additionally:
1. Generate reference files using `pose_datasets/save_pose_samples.py` and `save_motion_samples.py`
2. Set `get_eval_metrics=True` and fill in `eval_gt_pose_file` / `eval_gt_motion_file` in the config

### Stage 1 — Pose model (`multi_pose`)

Generates a single-frame multi-person SMPL pose from a text prompt.

```bash
cd diffusion
bash train_one_node.sh <num_gpus> configs_train/multi_pose.py
```

After training, note the path to your best checkpoint (e.g. `./outputs/multi_pose/.../ema_0.9999_XXXXXX.pt`).

### Stage 2 — Motion model (`pose2motion_2_stage`)

Generates a full motion sequence conditioned on a text prompt and the Stage 1 pose.

1. Open `diffusion/configs_train/pose2motion_2_stage.py`
2. Set `model.pose_model_ckpt` and `checkpoint_1st_stage` to your Stage 1 checkpoint path
3. Train:

```bash
cd diffusion
bash train_one_node.sh <num_gpus> configs_train/pose2motion_2_stage.py
```

### Multi-node training

Replace `train_one_node.sh` with `train_multi_node.sh <gpus_per_node> <num_nodes> <config>`.

---

## Sampling / Inference

After training both stages:

1. Open `diffusion/configs_sample/pose2motion_2_stage_webvid_frozen.py`
2. Set `checkpoint` (Stage 2) and `checkpoint_1st_stage` (Stage 1) to your trained checkpoint paths
3. Optionally set `cond_model_args.pose_path` and `cond_model_args.motion_uncond_path` for classifier guidance
4. Run:

```bash
cd diffusion
bash sample_one_node.sh <num_gpus> configs_sample/pose2motion_2_stage_webvid_frozen.py
```

Outputs are written to the directory specified by `out_dir` in the config (defaults to `None`, which writes alongside the checkpoint).

---

## Module Summary

## ```data_collection```

Code to gather (text, image, multi-person 3d poses) from static images. In the future, can consider expanding to video.

## ```diffusion```

Code for training a diffusion model using motion and pose data with text conditions.

## ```pose_corrector```

Code to shift or edit meshes from data or from model generations.

## ```pose_datasets```

Datasets for motion and pose in shared 3D SMPL format. Includes:

* ```amass_loader.py```: Single-person SMPL motion with text.
* ```laion_pose_loader.py```: Multi-person SMPL pose with text.
* ```joint_loader.py```: AMASS and LAION Pose data in common SMPL multiperson motion format.
* ```gta_loader.py```: (in progress) Single-person motion without text.
* ```mdm_loader.py```: HumanML3D-format of AMASS with text (for replication only, not development).

## ```smpl```

Code for SMPL-A mesh from the ROMP/BEV codebase, which adds meshes for infants and children to the standard SMPL model. For visualization and for pose editing. Should use this SML format for most cases (except for certain code inherited from MDM which uses the python ```smplx``` package.)

## ```viz```

Files to visualize motion and pose in shared format. Has the files:

* ```render.py```: Code for Python-Blender interface that allows for portable rendering without full Blender installation.
* ```viz_bev.py```: Visualize overlay of image and 3D SMPL mesh from BEV model predictions.
* ```viz_utils.py```: Code to save OBJ files and render Blender meshes from various pipelines.
