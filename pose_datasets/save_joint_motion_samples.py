import os

import numpy as np

from diffusion.dist_util import save_text_prompts


out_dir = '/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/amass_interhuman_v1/'

bank1 = '/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/amass_val_v1/data_samples_2753x61x158.npy'
text_file1 = '/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/amass_val_v1/data_samples_2753x61x158.txt'
bank2 = '/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/interhuman_v1/data_samples_7777x61x2x158.npy'
text_file2 = '/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/interhuman_v1/data_samples_7777x61x2x158.txt'

samples1 = np.load(bank1)
samples2 = np.load(bank2)

with open(text_file1, 'r') as f:
    text1 = f.readlines()
with open(text_file2, 'r') as f:
    text2 = f.readlines()

all_poses = np.concatenate((samples1[0:2048], samples2[0:1024, :, 0], samples2[0:1024, :, 1]), 0)
all_texts = text1[0:2048] + (2 * text2[0:1024])

out_path_npy = os.path.join(out_dir, f"data_samples_{'x'.join([str(x) for x in all_poses.shape])}.npy")
out_path_text = os.path.join(out_dir, f"data_samples_{'x'.join([str(x) for x in all_poses.shape])}.txt")
np.save(out_path_npy, all_poses)
save_text_prompts(out_path_text, all_texts)
