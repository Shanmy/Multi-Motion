from glob import glob
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset

from data_utils import convert_smpl_to_6d


class GTADataset(Dataset):
    def __init__(
        self,
        max_len=151,
        annot_path="/nfs/USRCSEA/IVA/Datasets/GTA-Human/gta_human/annotations",
        normalize=False,
        mean_path="/home/s9053168/code/multi-pose-diffusion/diffusion-v2/data_loaders/gta_mean.npy",  # TODO: put in public loc
        std_path="/home/s9053168/code/multi-pose-diffusion/diffusion-v2/data_loaders/gta_std.npy",  # TODO: put in public loc
    ):
        self.file_list = glob(annot_path + "/*.pkl")
        self.max_len = max_len
        self.normalize = normalize

        if self.normalize:
            try:
                self.mean = np.load(mean_path).reshape(1, 158).astype(np.float32)
            except:
                print('Warning: Unable to find mean file. Using zero mean instead.')
                self.mean = np.zeros(shape=[1, 158]).astype(np.float32)
            try:
                self.std = np.load(std_path).reshape(1, 158).astype(np.float32)
            except:
                print('Warning: Unable to find st dev file. Using one std dev instead.')
                self.std = np.ones(shape=[1, 158]).astype(np.float32)

    def __len__(self):
        return len(self.file_list)

    def _read_sample(self, idx):
        m_path = self.file_list[idx]
        with open(m_path, 'rb') as f:
            annot = pickle.load(f)
        pose = np.concatenate((annot['global_orient'], annot['body_pose']), axis=1)
        pose = convert_smpl_to_6d(pose).type(torch.float32)
        trans = torch.from_numpy(annot['transl']).type(torch.float32)
        pose_with_trans = torch.concat((pose, trans), axis=1)
        return pose_with_trans

    def __getitem__(self, idx):

        pose = self._read_sample(idx)  # [seq_len, 147]
        # manually append dummy betas. TODO: save and load these?
        betas = torch.zeros([pose.shape[0], 11])
        pose = torch.cat((betas, pose), dim=1)  # [seq_len, 158]

        # divide by mean and std
        if self.normalize:
            pose = (pose - self.mean) / self.std

        # add dummy poses to end of pose sequence so all sequences have same length
        pad_pose = np.zeros((self.max_len, pose.shape[1])).astype(np.float32)
        pad_pose[:pose.shape[0]] = pose[:min(pose.shape[0], self.max_len)]

        # mask indicating which parts of sequence are true poses and which are dummy poses
        mask = np.zeros([self.max_len]).astype(np.float32)
        mask[:pose.shape[0]] = 1
        assert mask.sum() != 0, "Zero length motion!"

        return pad_pose, {'mask': mask, 'lengths': np.sum(mask).astype(int)}


if __name__ == '__main__':

    import os
    from datetime import datetime
    from torch.utils.data import DataLoader

    from viz.viz_utils import viz_from_loader

    save_dir = 'out_gta_viz'
    save_dir = os.path.join(save_dir, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    batch_size = 10
    frameskip = 5  # spacing between frames of visualized samples, to speed up viz. set to 1 for full viz

    gta_dataset = GTADataset(normalize=False)
    gta_dataloader = DataLoader(gta_dataset, batch_size=batch_size)

    pose_sample = next(iter(gta_dataloader))

    # TODO: viz works but orientation of mesh looks incorrect
    viz_from_loader(
        poses=pose_sample[0],
        cond_dict=pose_sample[1],
        save_dir=save_dir,
        frameskip=frameskip,
        mean=None,
        std=None,
        unsqueeze_pose_dim=True,
        unsqueeze_motion_dim=False,
    )
