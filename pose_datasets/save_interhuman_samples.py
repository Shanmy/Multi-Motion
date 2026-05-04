import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion.dist_util import save_text_prompts
from pose_datasets.interhuman_loader import InterHumanDataset


out_dir = '/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/interhuman_v1/'
batch_size = 50

interhuman_dataset = InterHumanDataset(
    max_len=61,
)
interhuman_loader = DataLoader(interhuman_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)

all_poses = []
all_texts = []
for batch_idx, (pose, cond_dict) in enumerate(interhuman_loader):
    all_poses.append(pose)
    all_texts.extend(cond_dict['text'])
all_poses = torch.cat(all_poses, 0).numpy()
out_path_npy = os.path.join(out_dir, f"data_samples_{'x'.join([str(x) for x in all_poses.shape])}.npy")
out_path_text = os.path.join(out_dir, f"data_samples_{'x'.join([str(x) for x in all_poses.shape])}.txt")

np.save(out_path_npy, all_poses)
save_text_prompts(out_path_text, all_texts)
