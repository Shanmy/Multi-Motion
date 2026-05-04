import os

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("open_clip/src/training/")
sys.path.append("../pose_datasets/")
from clip_pose import setup_clmp
from amass_loader import AmassDataset

num_ckpts = 99
batch_size = 32
num_batches = 20
weight_path_stem = '/nfs/USRCSEA/IVA/Experiments/motion-diffusion/clip_pose/pose/2024_02_27-00_57_15-model_RN50-lr_0.0005-b_128-j_4-p_amp/checkpoints'

dataset = AmassDataset(max_len=61, split="val")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

for i in range(num_ckpts):
    weight_path = os.path.join(weight_path_stem, f'epoch_{i+1}.pt')
    clmp = setup_clmp('cuda', weight_path=weight_path)

    data_iterator = iter(dataloader)

    all_correct = 0
    for _ in range(num_batches):
        pose, cond = next(data_iterator)
        pose = pose.cuda()

        pose_features, text_features = clmp.encode_pose_and_text(pose, cond['text'])
        sim = pose_features @ text_features.t()

        correct = 0
        for j in range(sim.shape[0]):
            sim_bool = sim[j, j] >= sim[j]
            if sim_bool.prod() == 1:
                correct += 1
        all_correct += correct
    acc = all_correct / (num_batches * batch_size)
    print(f'Model ckpt {i + 1} acc: ', acc)
