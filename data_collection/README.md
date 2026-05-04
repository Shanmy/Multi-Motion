# Pose and Caption Dataset Repo (Updated 12/17/23)

This repo contains code to assemble a dataset of:

* Static triplets (image, human poses, caption)
* Dynamic triplets (video, human motions, caption) (future plans)

This data will be used to train generative models of human motion to animate virtual avatars according to natural language text descriptions.

## ROMP/BEV Setup

Run the command

```
bash setup_local.sh
```

## Run Commands

### Demo with Frisbee Data

```
python3 bev_blip2_image_folder.py
```

### Run LAION Data Collection

To collect a dataset of (image, pose, caption) triplets using LAION 400M images, run the command:

```
bash collect_data_with_log.sh NUM_GPUS NUM_NODES
```

where ```NUM_GPUS``` is the number of GPUs, ```NUM_NODES``` is the number of nodes.

### Visualize Random Samples from LAION Pose Data 

From the ```viz``` folder in the main directory, run the command:

```python3 viz_bev.py```

### Dataloader Demo

From the ```pose_datasetes``` folder in the main directory, run the command:

```python3 laion_pose_loader.py```

### Train CLIP Pose Model

From inside the folder ```open_clip```, run the command:

```
bash open_clip_pose.sh
```

This will train a contrastive (text, pose) model which can be used to select high-quality pairs.
