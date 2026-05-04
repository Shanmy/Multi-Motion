import os
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("open_clip/src/training/")
from clip_pose import setup_clpp
sys.path.append('../diffusion/diffusion/')
from script_util import create_data_loader

num_ckpts = 99
batch_size = 32
num_batches = 20
val = False
weight_path_stem = '/nfs/USRCSEA/IVA/Experiments/motion-diffusion/clip_pose/pose/2024_02_16-02_25_02-model_RN50-lr_0.0005-b_320-j_4-p_amp/checkpoints/'

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


for i in range(num_ckpts):
    weight_path = os.path.join(weight_path_stem, f'epoch_{i+1}.pt')
    clpp = setup_clpp('cuda', weight_path=weight_path)

    data_iterator = iter(dataloader)

    all_correct = 0
    for _ in range(num_batches):
        pose, cond = next(data_iterator)
        pose = pose.cuda()

        pose_features, text_features = clpp.encode_pose_and_text(pose, cond['text'])
        sim = pose_features @ text_features.t()

        correct = 0
        for j in range(sim.shape[0]):
            sim_bool = sim[j, j] >= sim[j]
            if sim_bool.prod() == 1:
                correct += 1
        all_correct += correct
    acc = all_correct / (num_batches * batch_size)
    print(f'Model ckpt {i + 1} acc: ', acc)
