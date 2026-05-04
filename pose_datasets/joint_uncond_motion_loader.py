import torch

from amass_loader import AmassDataset, AmassCompDataset
from laion_pose_loader import LaionPose
from interhuman_loader import InterHumanDataset
from webvid_motion_loader import WebVidMotionDataset
from data_utils import make_dataloader, get_pose_pad_batch


class JointMotionUncond:
    def __init__(
        self,
        batch_size,
        data_split=(0.5, 0.5),  # [amass, interhuman]
        shuffle=True,
        # args for motion dataset
        motion_max_len=61,
        amass_split="all",
        motion_use_dummy_betas=True,
        motion_use_dummy_hand_joints=True,
        amass_trans_aug_rad=None,
        interhuman_trans_aug_rad=None,
        # shared mean and std
        normalize=False,
        mean_path=None,
        std_path=None,
    ):

        self.motion_max_len = motion_max_len

        self.amass_batch_size = int(data_split[0] * batch_size)
        assert int(data_split[1] * batch_size) % 2 == 0
        self.interhuman_batch_size = int(data_split[1] * batch_size) // 2

        assert self.amass_batch_size > 0 or self.interhuman_batch_size > 0
        assert self.amass_batch_size + (2 * self.interhuman_batch_size) == batch_size

        # initial dataloaders

        if self.amass_batch_size > 0:
            self.amass_dataset = AmassDataset(
                max_len=motion_max_len,
                split=amass_split,
                use_dummy_betas=motion_use_dummy_betas,
                use_dummy_hand_joints=motion_use_dummy_hand_joints,
                use_dummy_text=True,
                normalize=normalize,
                mean_path=mean_path,
                std_path=std_path,
                trans_aug_rad=amass_trans_aug_rad,
            )
            self.amass_loader = make_dataloader(
                self.amass_dataset, batch_size=self.amass_batch_size, shuffle=shuffle
            )

        if self.interhuman_batch_size > 0:
            self.interhuman_dataset = InterHumanDataset(            
                max_len=motion_max_len,
                use_dummy_betas=motion_use_dummy_betas,
                use_dummy_hand_joints=motion_use_dummy_hand_joints,
                use_dummy_text=True,
                normalize=normalize,
                mean_path=mean_path,
                std_path=std_path,
                trans_aug_rad=interhuman_trans_aug_rad,
            )
            self.interhuman_loader = make_dataloader(
                self.interhuman_dataset, batch_size=self.interhuman_batch_size, shuffle=shuffle
            )

    def __iter__(self):
        return self

    def _process_amass_pose(self, pose_amass):
        pose_amass = get_pose_pad_batch(
                        pose_amass,
                        lengths=[self.motion_max_len] * pose_amass.shape[0],
                        max_len=self.motion_max_len,
                    )
        return pose_amass

    def _process_interhuman_pose(self, pose_interhuman):
        pose_interhuman = get_pose_pad_batch(
                        pose_interhuman,
                        lengths=[self.motion_max_len] * pose_interhuman.shape[0],
                        num_poses=[2] * pose_interhuman.shape[0],
                        max_len=self.motion_max_len,
                        max_poses=2,
                    )
        pose_interhuman = torch.cat((pose_interhuman[:, :, 0], pose_interhuman[:, :, 1]), 0)

        return pose_interhuman

    def _process_interhuman_cond_dict(self, interhuman_cond_dict):
        cond_dict_out = {
            'text': interhuman_cond_dict['text'] + interhuman_cond_dict['text'],
            'lengths': torch.cat((interhuman_cond_dict['lengths'], interhuman_cond_dict['lengths']), 0),
        }
        return cond_dict_out

    def _merge_conditions(
        self, cond_dict_list, shared_keys=('text', 'lengths',)
    ):

        # merge conditions
        cond_dict_merge = {
            k: (
                torch.cat(([cond_dict[k] for cond_dict in cond_dict_list]), 0)
                if torch.is_tensor(cond_dict_list[0][k])
                else sum([cond_dict[k] for cond_dict in cond_dict_list], [])
            )
            for k in shared_keys
        }

        return cond_dict_merge

    def __next__(self):
        # get separate poses from each type of data
        # add pose dim and dummy poses to amass sample.
        # common shape [*, motion_max_length, max_poses, 158]

        all_poses = []
        all_cond_dict = []
        if self.amass_batch_size > 0:
            pose_amass, cond_dict_amass = next(self.amass_loader)
            pose_amass = self._process_amass_pose(pose_amass)
            all_poses.append(pose_amass)
            all_cond_dict.append(cond_dict_amass)
        if self.interhuman_batch_size > 0:
            pose_interhuman, cond_dict_interhuman = next(self.interhuman_loader)
            pose_interhuman = self._process_interhuman_pose(pose_interhuman)
            cond_dict_interhuman = self._process_interhuman_cond_dict(cond_dict_interhuman)
            all_poses.append(pose_interhuman)
            all_cond_dict.append(cond_dict_interhuman)

        if len(all_poses) == 0:
            raise ValueError('Either amass batch size or laion pose batch size or interhuman batch size must be positive.')
        pose_out = torch.cat(all_poses, dim=0)
        # merge conditions into single dictionary
        cond_dict = self._merge_conditions(all_cond_dict)

        return pose_out, cond_dict


if __name__ == '__main__':

    import os
    from datetime import datetime

    from viz.viz_utils import viz_from_loader

    save_dir = 'out_joint_viz'
    save_dir = os.path.join(save_dir, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))

    batch_size = 40
    data_split = (0.5, 0.5)
    frameskip = 10

    joint_loader = JointMotionUncond(
        batch_size=batch_size,
        data_split=data_split,
    )

    pose_sample = next(iter(joint_loader))

    viz_from_loader(
        poses=pose_sample[0],
        cond_dict=pose_sample[1],
        save_dir=save_dir,
        frameskip=frameskip,
        mean_path=None,
        std_path=None,
        unsqueeze_pose_dim=True,
        unsqueeze_motion_dim=False,
    )
