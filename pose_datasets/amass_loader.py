import os
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from pose_datasets.data_utils import (
    list_files_recursively,
    make_dataloader,
    process_translations,
    convert_smpl_to_6d,
    get_pose_pad,
    pose_random_rotation,
    truncate_frames,
)


class AmassDataset(Dataset):
    def __init__(
        self,
        max_len=120,
        motion_path="/nfs/USRCSEA/IVA/Datasets/AMASS/amass_data/",
        text_path="/nfs/USRCSEA/IVA/Datasets/HumanML3D/texts/",
        split="all",
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
    ):

        self.normalize = normalize
        self.max_len = max_len
        self.use_dummy_betas = use_dummy_betas
        self.use_dummy_hand_joints = use_dummy_hand_joints
        self.use_dummy_text = use_dummy_text

        self.process_height = process_height
        self.trans_aug_rad = trans_aug_rad
        # set to True to center the translation of first frame. False will center across all frames.
        self.center_first_frame = center_first_frame
        self.randomly_rotate_pose = randomly_rotate_pose

        self.motion_path = motion_path
        # list all npz candidate files
        self.file_list_npz = list_files_recursively(self.motion_path, file_types=('npz',))
        self.remove_shape_files()
        self.file_list_npz.sort()

        if not self.use_dummy_text:
            self.text_path = text_path
            if split == 'all':
                self.text_mapping_file = text_mapping_file_all
            elif split == 'train':
                self.text_mapping_file = text_mapping_file_train
            elif split == 'val':
                self.text_mapping_file = text_mapping_file_val
            else:
                return ValueError("Invalid 'split' ('all', 'train', or 'val').")
            # process text annotations for each file
            self.file_list, self.file_list_text, self.start_frames, self.end_frames = self.process_text_files()
        else:
            self.file_list = self.file_list_npz
            self.file_list_text = None
            self.start_frames = None
            self.end_frames = None

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

    def remove_shape_files(self, shape_file_name="shape.npz"):
        for file_name in self.file_list_npz:
            if file_name.endswith(shape_file_name):
                self.file_list_npz.remove(file_name)

    def get_npz_file_name(self, index_csv_str):
        str_out = "/".join(index_csv_str.split("/")[2:])
        str_out = str_out.split(".")[0] + ".npz"
        return str_out

    def process_text_files(self):

        mapping_data = pd.read_csv(self.text_mapping_file)
        motion_files = list(mapping_data.source_path)
        text_files = list(mapping_data.text_file)
        start_frames_all = list(mapping_data.start_frame)
        end_frames_all = list(mapping_data.end_frame)

        # sort all lists according to motion alphabetical file order
        # https://stackoverflow.com/questions/11601961/sorting-multiple-lists-based-on-a-single-list-in-python
        zipped = zip(motion_files, text_files, start_frames_all, end_frames_all)
        zipped = sorted(zipped)
        motion_files, text_files, start_frames_all, end_frames_all = zip(*zipped)

        file_list = []
        file_list_text = []
        start_frames = []
        end_frames = []

        motion_file_idx = 0

        for npz_file_idx, file_name in enumerate(self.file_list_npz):
            while motion_files[motion_file_idx] in file_name:
                file_list.append(file_name)
                file_list_text.append(os.path.join(self.text_path, text_files[motion_file_idx]))
                start_frames.append(start_frames_all[motion_file_idx])
                end_frames.append(end_frames_all[motion_file_idx])
                motion_file_idx += 1

        assert len(file_list) == len(file_list_text) == len(start_frames) == len(end_frames)

        return file_list, file_list_text, start_frames, end_frames

    def __len__(self):
        return len(self.file_list)

    def _prepare_sample(self, idx, ex_fps=20):
        # adapted from https://github.com/EricGuo5513/HumanML3D/blob/main/raw_pose_processing.ipynb

        source_path = self.file_list[idx]
        data = np.load(source_path)

        # TODO: betas probably in the wrong format. fix?
        betas = data['betas']
        thetas = data['poses']
        trans = data['trans']
        fps = data['mocap_framerate']
        num_frames = trans.shape[0]

        if fps > ex_fps:
            downsample_ratio = int(fps / ex_fps)
        else:
            downsample_ratio = 1

        betas = betas[0:11].reshape(1, 11).repeat(num_frames, 0)

        # keep first 24 joints (22 body joints plus 2 hand joints)
        thetas = thetas.reshape(-1, 52, 3)[:, 0:24].reshape(-1, 24 * 3)
        # convert to 6d representation
        thetas_6d = convert_smpl_to_6d(thetas).numpy()

        pose = np.concatenate((betas, thetas_6d, trans), 1)
        pose = pose[::downsample_ratio]

        # inhereted from humanml3d preprocessing. needed to align text captions.
        if 'Eyes_Japan_Dataset' in source_path:
            if pose.shape[0] > (3 * ex_fps):
                pose = pose[(3 * ex_fps):]
        if 'MPI_HDM05' in source_path:
            if pose.shape[0] > (3 * ex_fps):
                pose = pose[(3 * ex_fps):]
        if 'TotalCapture' in source_path:
            if pose.shape[0] > (1 * ex_fps):
                pose = pose[(1 * ex_fps):]
        if 'MPI_Limits' in source_path:
            if pose.shape[0] > (1 * ex_fps):
                pose = pose[(1 * ex_fps):]
        if 'Transitions_mocap' in source_path:
            if pose.shape[0] > (0.5 * ex_fps):
                pose = pose[int(0.5 * ex_fps):]

        if self.start_frames is not None and self.end_frames is not None:
            start_frame = self.start_frames[idx]
            end_frame = self.end_frames[idx]
            pose = pose[start_frame:end_frame]

        return pose

    def __getitem__(self, idx):

        # Load pose, [f_num x 158]
        pose = self._prepare_sample(idx)
        pose = pose.astype(np.float32)

        pose = truncate_frames(pose, self.max_len)
        lengths = min(pose.shape[0], self.max_len)
        pose = pose[:lengths]

        # TODO: add random cropping of sequence instead of only getting start of sequence?
        if self.use_dummy_betas:
            # set the beta elements to 0
            pose[:, :11] = 0
        if self.use_dummy_hand_joints:
            # set hand joint pose id matrix (since these were not collected by BEV)
            pose[:, -15:-3] = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])

        # center and augment translations
        pose[:, -3:] = process_translations(
                            pose[:, -3:],
                            center_first_frame=self.center_first_frame,
                            single_motion_mode=True,
                            trans_aug_rad=self.trans_aug_rad,
                            process_height=self.process_height,
                        )

        if self.randomly_rotate_pose:
            pose = pose_random_rotation(pose)

        # divide by mean and std
        if self.normalize:
            pose = (pose - self.mean) / self.std

        pad_pose = get_pose_pad(pose, max_len=self.max_len, return_npy=True)

        if not self.use_dummy_text:
            # Load text, randomized
            with open(self.file_list_text[idx]) as f:
                captions = []
                for line in f.readlines():
                    line_split = line.strip().split('#')[0]
                    captions.append(line_split)
                text = random.choice(captions)
                text = text.lower()
                if not text.endswith('.'):
                    # always end in period (helpful for making the joint captions)
                    text += '.'
        else:
            # empty string as "unconditional" caption
            text = ''
        lengths = np.array(lengths)

        return pad_pose, {'text': text, 'lengths': lengths}


class AmassCompDataset:
    def __init__(
        self,
        batch_size,
        shuffle=True,
        comp_probs=(0.1, 0.1, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0),
        max_len=120,
        max_poses=10,
        motion_path="/nfs/USRCSEA/IVA/Datasets/AMASS/amass_data/",
        text_path="/nfs/USRCSEA/IVA/Datasets/HumanML3D/texts/",
        split="all",
        text_mapping_file_train="/nfs/USRCSEA/IVA/Datasets/AMASS/index_train.csv",
        text_mapping_file_val="/nfs/USRCSEA/IVA/Datasets/AMASS/index_val.csv",
        text_mapping_file_all="/nfs/USRCSEA/IVA/Datasets/AMASS/index.csv",
        use_dummy_betas=True,  # whether to overwrite betas with 0
        use_dummy_hand_joints=False,  # whether to overwite hand joint poses with 0 vector
        use_dummy_text=True,  # whether to overwrite text from dataset with empty string
        normalize=False,
        mean_path=None,
        std_path=None,
        center_first_frame=True,
        randomly_rotate_pose=True,
        process_height=True,
        trans_aug_rad=None,
        trans_aug_rad_single=(0.75, 0.75, 0),  # augmentations for individuals to spread out poses from origin
    ):

        # probability for different number of people in composite motion
        self.comp_probs = comp_probs
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_poses = max_poses
        self.center_first_frame = center_first_frame
        self.randomly_rotate_pose = randomly_rotate_pose
        self.process_height = process_height
        self.trans_aug_rad = trans_aug_rad
        assert len(comp_probs) <= self.max_poses

        self.amass_dataset = AmassDataset(
            max_len=max_len,
            motion_path=motion_path,
            text_path=text_path,
            split=split,
            text_mapping_file_train=text_mapping_file_train,
            text_mapping_file_val=text_mapping_file_val,
            text_mapping_file_all=text_mapping_file_all,
            use_dummy_betas=use_dummy_betas,  # whether to overwrite betas with 0
            use_dummy_hand_joints=use_dummy_hand_joints,  # whether to overwite hand joint poses with 0 vector
            use_dummy_text=use_dummy_text,  # whether to overwrite text from dataset with empty string
            center_first_frame=center_first_frame,
            randomly_rotate_pose=randomly_rotate_pose,
            process_height=False,
            trans_aug_rad=trans_aug_rad_single,
            normalize=normalize,
            mean_path=mean_path,
            std_path=std_path,
        )
        # default batch size set to a large number, poses allocated dynamically in __next__ 
        self.amass_dataloader = make_dataloader(self.amass_dataset, batch_size=100, shuffle=shuffle)

    def __iter__(self):
        return self

    def _get_pose_batch(self, total_poses):
        num_poses = 0
        poses = []
        lengths = []
        text = []
        while num_poses < total_poses:
            pose, cond_dict = next(self.amass_dataloader)
            poses.append(pose)
            lengths.append(cond_dict['lengths'])
            text += cond_dict['text']
            num_poses += pose.shape[0]
        poses = torch.cat(poses, 0)
        lengths = torch.cat(lengths, 0)
        return poses, lengths, text

    def _organize_pose_batch(self, pose, lengths, num_poses, text):

        pose_out = []
        lengths_out = []
        text_out = []
        start_idx = 0
        for idx in range(self.batch_size):
            end_idx = start_idx + num_poses[idx]

            length_batch = lengths[start_idx:end_idx].min()
            lengths_out.append(length_batch.view(1))
            text_out.append(' '.join(text[start_idx:end_idx]))

            pose_batch = pose[start_idx:end_idx]
            # frames to leading dim, number of poses to next dim
            pose_batch = pose_batch.permute(1, 0, 2).contiguous()
            # pad along frame and pose dims
            pose_batch = get_pose_pad(pose_batch, max_len=self.max_len, max_poses=self.max_poses)

            pose_out.append(pose_batch)
            start_idx = end_idx

        pose_out = torch.stack(pose_out, 0).numpy()

        # center and augment translations of group
        for idx in range(pose_out.shape[0]):
            trans_out = process_translations(
                                pose_out[idx, :lengths[idx], :num_poses[idx], -3:],
                                center_first_frame=self.center_first_frame,
                                trans_aug_rad=self.trans_aug_rad,
                                process_height=self.process_height,
                            )
            pose_out[idx, :lengths[idx], :num_poses[idx], -3:] = trans_out

            if self.randomly_rotate_pose:
                pose_out[idx] = pose_random_rotation(pose_out[idx])

        pose_out = torch.tensor(pose_out)
        lengths_out = torch.cat(lengths_out, 0)

        return pose_out, lengths_out, text_out

    def __next__(self):
        # get the number of composite poses to yield
        num_poses = np.random.choice(np.arange(len(self.comp_probs)), self.batch_size, p=self.comp_probs) + 1
        num_poses = torch.tensor(num_poses)
        total_poses = num_poses.sum().item()

        # get batch of poses from the single-person amass iterator
        pose_batch, lengths_batch, text_batch = self._get_pose_batch(total_poses)

        # change unstructed batch into multi person motion composites
        pose, lengths, text = self._organize_pose_batch(pose_batch, lengths_batch, num_poses, text_batch)

        return pose, {'lengths': lengths, 'num_poses': num_poses, 'text': text}


# Example usage:
if __name__ == '__main__':

    from datetime import datetime
    from torch.utils.data import DataLoader

    from viz.viz_utils import viz_from_loader

    COMPOSITE = True

    save_dir = 'out_amass_viz'
    save_dir = os.path.join(save_dir, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    batch_size = 10
    frameskip = 10  # spacing between frames of visualized samples, to speed up viz. set to 1 for full viz

    if COMPOSITE:
        amass_dataloader = AmassCompDataset(
                                normalize=False,
                                batch_size=batch_size,
                                trans_aug_rad=None,
                                shuffle=True,
                                use_dummy_text=True,
                            )
    else:
        amass_dataset = AmassDataset(
                            normalize=False,
                            trans_aug_rad=None,
                            randomly_rotate_pose=True,
                        )
        amass_dataloader = DataLoader(amass_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    amass_iterator = iter(amass_dataloader)

    pose_sample = next(amass_iterator)

    viz_from_loader(
        poses=pose_sample[0],
        cond_dict=pose_sample[1],
        save_dir=save_dir,
        frameskip=frameskip,
        mean_path=None,
        std_path=None,
        unsqueeze_pose_dim=(not COMPOSITE),
        unsqueeze_motion_dim=False,
    )
