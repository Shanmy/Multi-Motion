# Pose Datasets (Updated 12/17/23)

## Data Formats

All data code uses one of the formats below:

* (num_frames, 158) for single-person motion data
* (num_poses, 158) for multi-person pose data
* (num_frames, num_poses, 158) for multi-person motion data

The 158-dim pose format is used in the dataloader and training code, and breaks down into the follow parts:

* First 11 dims: SMPL Betas/Shape parameters
* Next 144 dims: 6D format of SMPL Thetas/Pose parameters (24 joints x 6 dims per joint)
* Next 3 dims: 3D global translation.

## Dataloaders

Running the ```.py``` file for each dataloader will draw a batch, save the ```.obj``` files, and save a rendering visualization. For example:

```
python3 laion_pose_loader.py
```

The dataloader files are:

* ```amass_loader.py```: Single-person SMPL motion with text.
* ```laion_pose_loader.py```: Multi-person SMPL pose with text.
* ```joint_loader.py```: AMASS and LAION Pose data in common SMPL multiperson motion format.
* ```gta_loader.py```: (in progress) Single-person motion without text.
* ```mdm_loader.py```: HumanML3D-format of AMASS with text (for replication only, not development). No visualization support.
