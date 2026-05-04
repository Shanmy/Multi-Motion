import os
import random
from glob import glob

import numpy as np
import pickle as pkl

from torch.utils.data import Dataset

from data_utils import (
    convert_smpl_to_6d,
    process_translations,
    get_pose_pad,
    truncate_frames,
    pose_random_rotation,
)


class InterHumanDataset(Dataset):
    def __init__(
        self,
        max_len=61,
        motion_path="/nfs/USRCSEA/IVA/Datasets/InterHuman/motions/",
        text_path="/nfs/USRCSEA/IVA/Datasets/InterHuman/annots/",
        use_dummy_betas=True,
        use_dummy_hand_joints=True,
        use_dummy_text=False,
        normalize=False,
        mean_path=None,
        std_path=None,
        center_first_frame=True,
        randomly_rotate_pose=True,
        trans_aug_rad=None,
    ):

        self.motion_path = motion_path
        self.all_motions = glob(os.path.join(self.motion_path, '*.pkl'))
        self.remove_unused_motions()
        self.text_path = text_path
        self.normalize = normalize
        self.max_len = max_len
        self.use_dummy_betas = use_dummy_betas
        self.use_dummy_hand_joints = use_dummy_hand_joints
        self.use_dummy_text = use_dummy_text

        self.trans_aug_rad = trans_aug_rad
        # set to True to center the translation of first frame. False will center across all frames.
        self.center_first_frame = center_first_frame
        self.randomly_rotate_pose = randomly_rotate_pose

        if self.normalize:
            try:
                self.mean = np.load(mean_path).reshape(1, 1, 158).astype(np.float32)
            except:
                print('Warning: Unable to find mean file. Using zero mean instead.')
                self.mean = np.zeros(shape=[1, 1, 158]).astype(np.float32)
            try:
                self.std = np.load(std_path).reshape(1, 1, 158).astype(np.float32)
            except:
                print('Warning: Unable to find st dev file. Using one std dev instead.')
                self.std = np.ones(shape=[1, 1, 158]).astype(np.float32)

    def remove_unused_motions(self):
        # edit motion file list to exclude duplicates with (1) in file name and certain files which are empty
        self.all_motions = [
            file_name for file_name in self.all_motions
            if ('(1)' not in file_name) and 
            (not (file_name.endswith('3945.pkl') or file_name.endswith('4106.pkl')))
        ]

    def __len__(self):
        return len(self.all_motions)

    def _process_one_person(self, pose):
        # TODO: betas probably in the wrong format. fix?
        beta = pose['betas']  # (10)

        trans = pose['trans']  # (num_frames, 3)
        thetas_root = pose['root_orient']  # (num_frames, 3)
        thetas_body = pose['pose_body']  # (num_frames, 63)
        num_frames = trans.shape[0]

        thetas = np.concatenate((thetas_root, thetas_body, np.zeros((num_frames, 6))), axis=1)
        thetas_6d = convert_smpl_to_6d(thetas).numpy()

        smpl = np.zeros((num_frames, 158))
        smpl[:, :10] = beta
        smpl[:, 11:-3] = thetas_6d
        smpl[:, -3:] = trans

        return smpl

    def _prepare_sample(self, idx, exp_fps=20):
        with open(self.all_motions[idx], 'rb') as f:
            data = pkl.load(f)

        fps = data['mocap_framerate']
        p1 = data['person1']
        p2 = data['person2']

        if fps > 20:
            downsample_ratio = int(fps / exp_fps)
        else:
            downsample_ratio = 1

        smpl1 = self._process_one_person(p1)[::downsample_ratio]
        smpl2 = self._process_one_person(p2)[::downsample_ratio]

        smpl1 = smpl1.reshape(-1, 1, smpl1.shape[-1])
        smpl2 = smpl2.reshape(-1, 1, smpl2.shape[-1])
        smpl_out = np.concatenate((smpl1, smpl2), 1)

        return smpl_out

    def __getitem__(self, idx):

        # Load pose, [num_frames, 2, 158]
        pose = self._prepare_sample(idx)
        pose = pose.astype(np.float32)

        pose = truncate_frames(pose, self.max_len)
        lengths = min(pose.shape[0], self.max_len)
        pose = pose[:lengths]

        pose = pose.astype(np.float32)
        if self.use_dummy_betas:
            # set the beta elements to 0
            pose[:, :, :11] = 0
        if self.use_dummy_hand_joints:
            # set hand joint pose id matrix (since these were not collected by BEV)
            pose[:, :, -15:-3] = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])

        # center and augment translations
        pose[:, :, -3:] = process_translations(
                                pose[:, :, -3:],
                                center_first_frame=self.center_first_frame,
                                trans_aug_rad=self.trans_aug_rad
                            )

        if self.randomly_rotate_pose:
            pose = pose_random_rotation(pose)

        # divide by mean and std
        if self.normalize:
            pose = (pose - self.mean) / self.std

        # add dummy poses to end of pose sequence so all sequences have same length
        pad_pose = get_pose_pad(pose, max_len=self.max_len, max_poses=2, return_npy=True)

        # Load text, randomized
        if self.use_dummy_text:
            text = ''
        else:
            sample_txt_file = self.all_motions[idx].split("/")[-1]
            sample_txt_file = sample_txt_file.split(".")[0] + ".txt"
            with open(os.path.join(self.text_path, sample_txt_file)) as f:
                captions = []
                for line in f.readlines():
                    line_split = line.strip().split('#')[0]
                    captions.append(line_split)
                text = random.choice(captions)
                text = text.lower()
        lengths = np.array(lengths)
        num_poses = np.array(2)

        return pad_pose, {'text': text, 'lengths': lengths, 'num_poses': num_poses}


if __name__ == "__main__":
    
    from datetime import datetime
    from torch.utils.data import DataLoader

    from viz.viz_utils import viz_from_loader

    save_dir = 'out_interhuman_viz'
    save_dir = os.path.join(save_dir, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    batch_size = 8
    frameskip = 10  # spacing between frames of visualized samples, to speed up viz. set to 1 for full viz

    interhuman_dataset = InterHumanDataset(
                            normalize=False,
                            randomly_rotate_pose=True,
                            max_num_trunc_frames=40,
                        )
    interhuman_dataloader = DataLoader(
                                interhuman_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=1,
                            )

    pose_sample = next(iter(interhuman_dataloader))

    viz_from_loader(
        poses=pose_sample[0],
        cond_dict=pose_sample[1],
        save_dir=save_dir,
        frameskip=frameskip,
        mean_path=None,
        std_path=None,
        unsqueeze_pose_dim=False,
        unsqueeze_motion_dim=False,
    )
