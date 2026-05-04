# Pose and Motion Visualization (Updated 12/17/23)

## BEV Rendering of LAION Pose Data

To visualize BEV renderings of LAION Pose samples overlaid on the image they were collected from, run:
```python3 viz_bev.py```.
This is a basic tool for checking the quality of the LAION Pose data which displays the data in the context where it was observed.

## Blender Rendering

Two kinds of blender rendering are supported. For full Blender use, the viz code output will save ```obj``` files which can be downloaded to a local device which has Blender. There is also a Python-Blender interface that can display Blender rendering on the local server.

See ```README_blender_setup_instructions.txt``` for setup info. NOTE: This setup has already been performed on the local server and should work for everyone already.

## Format for Visualization

All visualization code assumes that pose samples have one of the formats below:

* (num_frames, 158) for single-person motion data (use flag ```unsqueeze_pose_dim``` in viz functions)
* (num_poses, 158) for pose data (use flag ```unsqueeze_motion_dim``` in viz functions)
* (num_frames, num_poses, 158) for multi-person motion data

The 158-dim pose format is used in the dataloader and training code, and breaks down into the follow parts:

* First 11 dims: SMPL Betas/Shape parameters
* Next 144 dims: 6D format of SMPL Thetas/Pose parameters (24 joints x 6 dims per joint)
* Next 3 dims: 3D global translation.
