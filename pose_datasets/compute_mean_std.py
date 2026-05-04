import os

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from laion_pose_loader import LaionPose
from amass_loader import AmassDataset, AmassCompDataset
from interhuman_loader import InterHumanDataset
from webvid_motion_loader import WebVidMotionDataset
from data_utils import remove_pad_poses


def compute_mean_std_from_loader(dataloader, out_dir='stats/', max_poses=10000, save=True, eps=1e-5):

    all_data = []
    num_poses_collected = 0

    for poses, cond in dataloader:
        pose = poses[0]
        num_frames = cond['lengths'][0] if 'lengths' in cond.keys() else None
        num_poses = cond['num_poses'][0] if 'num_poses' in cond.keys() else None
        pose = remove_pad_poses(pose, num_frames=num_frames, num_poses=num_poses)
        pose = pose.contiguous().view(-1, pose.shape[-1])
        num_poses_collected += pose.shape[0]
        all_data.append(pose)
        if num_poses_collected >= max_poses:
            break

    all_data = np.concatenate(all_data, axis=0)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0) + eps

    if save:
        np.save(os.path.join(out_dir, 'smpl_158_mean.npy'), mean)
        np.save(os.path.join(out_dir, 'smpl_158_std.npy'), std)

    return mean, std, all_data


def compute_mean_std_laion_pose(out_dir='stats_laion_pose/', save=True, batch_size=100):
    dataloader = LaionPose(
        normalize=False,
        batch_size=batch_size,
        resampled=True,
        concat_vars=True,
        yield_cond_dict=True,
        #trans_aug_rad=None,
        trans_aug_rad=(0.45, 0.45, 0.0),
    )
    mean, std, all_data = compute_mean_std_from_loader(dataloader, out_dir=out_dir, save=save)
    return mean, std, all_data


def compute_mean_std_amass(out_dir='stats_amass/', save=True, batch_size=100):
    dataset = AmassDataset(
                normalize=False,
                mean_path=None,
                std_path=None,
                trans_aug_rad=None,
                #trans_aug_rad=(0.7, 0.9, 0.12),
            )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    mean, std, all_data = compute_mean_std_from_loader(dataloader, out_dir=out_dir, save=save)
    return mean, std, all_data


def compute_mean_std_amass_comp(out_dir='stats_amass_comp/', save=True, batch_size=100):
    dataloader = AmassCompDataset(
        batch_size=batch_size,
        shuffle=True,
        normalize=False,
        trans_aug_rad=None,
        #trans_aug_rad=(0.7, 0.9, 0.12),
    )
    mean, std, all_data = compute_mean_std_from_loader(dataloader, out_dir=out_dir, save=save)
    return mean, std, all_data


def compute_mean_std_interhuman(out_dir='stats_interhuman/', save=True, batch_size=100):
    dataset = InterHumanDataset(
                    normalize=False,
                    trans_aug_rad=None,
                    #trans_aug_rad=(0.25, 0.9, 0.12),
                )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    mean, std, all_data = compute_mean_std_from_loader(dataloader, out_dir=out_dir, save=save)
    return mean, std, all_data


def compute_mean_std_webvid_motion(out_dir='stats_webvid_motion', save=True, batch_size=100):
    dataset = WebVidMotionDataset(
                    normalize=False,
                    trans_aug_rad=None,
                    #trans_aug_rad=(0.0, 0.0, 0.0),
                )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    mean, std, all_data = compute_mean_std_from_loader(dataloader, out_dir=out_dir, save=save)
    return mean, std, all_data


if __name__ == '__main__':

    mean_laion, std_laion, data_laion = compute_mean_std_laion_pose(save=False)
    mean_amass, std_amass, data_amass = compute_mean_std_amass(save=False)
    mean_amass_comp, std_amass_comp, data_amass_comp = compute_mean_std_amass_comp(save=False)
    mean_interhuman, std_interhuman, data_interhuman = compute_mean_std_interhuman(save=False)
    mean_webvid, std_webvid, data_webvid = compute_mean_std_webvid_motion(save=False)

    print(f'Laion Trans mean/std: {mean_laion[-3:]}/{std_laion[-3:]}')
    print(f'Amass Trans mean/std: {mean_amass[-3:]}/{std_amass[-3:]}')
    print(f'Comp Trans mean/std: {mean_amass_comp[-3:]}/{std_amass_comp[-3:]}')
    print(f'Interhuman Trans mean/std: {mean_interhuman[-3:]}/{std_interhuman[-3:]}')
    print(f'Webvid Trans mean/std: {mean_webvid[-3:]}/{std_webvid[-3:]}')

    for i in range(3):
        plt.hist(data_laion[:, -(3 - i)], alpha=0.5, density=True, label='laion')
        plt.hist(data_amass[:, -(3 - i)], alpha=0.5, density=True, label='amass')
        plt.hist(data_amass_comp[:, -(3 - i)], alpha=0.5, density=True, label='amass_comp')
        plt.hist(data_interhuman[:, -(3 - i)], alpha=0.5, density=True, label='interhuman')
        plt.hist(data_webvid[:, -(3 - i)], alpha=0.5, density=True, label='webvid')
        plt.legend(loc='upper right')
        plt.savefig(f'out_joint_viz/trans_hist_{i}.png')
        plt.close()
