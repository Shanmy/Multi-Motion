import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion.dist_util import save_text_prompts
from pose_datasets.amass_loader import AmassDataset


out_dir = '/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/amass_val_v2'
batch_size = 50

amass_dataset = AmassDataset(
    max_len=61,
    motion_path="/nfs/USRCSEA/IVA/Datasets/AMASS/amass_data/",
    text_path="/nfs/USRCSEA/IVA/Datasets/HumanML3D/texts/",
    split="val",
    text_mapping_file_train="/nfs/USRCSEA/IVA/Datasets/AMASS/index_train.csv",
    text_mapping_file_val="/nfs/USRCSEA/IVA/Datasets/AMASS/index_val.csv",
    text_mapping_file_all="/nfs/USRCSEA/IVA/Datasets/AMASS/index.csv",
    use_dummy_betas=True,  # whether to overwrite betas with 0
    use_dummy_hand_joints=True,  # whether to overwite hand joint poses with 0 vector
    use_dummy_text=False,  # whether to use empty string as text condition
    normalize=False,
    mean_path=None,
    std_path=None,
    center_first_frame=True,
    randomly_rotate_pose=True,
    process_height=True,
    trans_aug_rad=None,
)
amass_loader = DataLoader(amass_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)

all_poses = []
all_texts = []
for batch_idx, (pose, cond_dict) in enumerate(amass_loader):
    all_poses.append(pose)
    all_texts.extend(cond_dict['text'])
all_poses = torch.cat(all_poses, 0).numpy()
out_path_npy = os.path.join(out_dir, f"data_samples_{'x'.join([str(x) for x in all_poses.shape])}.npy")
out_path_text = os.path.join(out_dir, f"data_samples_{'x'.join([str(x) for x in all_poses.shape])}.txt")

np.save(out_path_npy, all_poses)
save_text_prompts(out_path_text, all_texts)
