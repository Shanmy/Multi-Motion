import os
from omegaconf import OmegaConf

import numpy as np
import torch

from diffusion.script_util import create_data_loader
from diffusion.dist_util import save_text_prompts


out_dir = '/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/laion_pose_v4'
total_samples = 10000
batch_size = 50
val = False

dataset_args_laion = dict(
    data_type='laion_pose',
    pose_use_dummy_text=False,
    pose_use_dummy_betas=True,
    normalize=False,
    mean_path=None,
    std_path=None,
    process_trans=True,
    randomly_rotate_pose=True,
    trans_aug_rad=None,
    align_pose=False,
    val=val,
)
dataset_args_laion = OmegaConf.create(dataset_args_laion)
dataloader = create_data_loader(dataset_args_laion, batch_size=batch_size)

all_poses = []
all_texts = []
for _ in range(total_samples // batch_size):
    pose, cond_dict = next(dataloader)
    all_poses.append(pose)
    all_texts.extend(cond_dict['text'])
all_poses = torch.cat(all_poses, 0).numpy()
out_path_npy = os.path.join(out_dir, f"data_samples_{'x'.join([str(x) for x in all_poses.shape])}.npy")
out_path_text = os.path.join(out_dir, f"data_samples_{'x'.join([str(x) for x in all_poses.shape])}.txt")

np.save(out_path_npy, all_poses)
save_text_prompts(out_path_text, all_texts)
