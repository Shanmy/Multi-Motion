import torch

from pose_datasets.amass_loader import AmassDataset, AmassCompDataset
from pose_datasets.laion_pose_loader import LaionPose
from pose_datasets.interhuman_loader import InterHumanDataset
from pose_datasets.webvid_motion_loader import WebVidMotionDataset
from pose_datasets.data_utils import make_dataloader, get_pose_pad_batch


class JointMotionPose:
    def __init__(
        self,
        batch_size,
        data_split=(0.5, 0.2, 0.2, 0.1, 0.0),  # [pose, amass, interhuman, webvid motion, amass composite]
        shuffle=True,
        # args for motion dataset
        motion_max_len=120,
        amass_path="/nfs/USRCSEA/IVA/Datasets/AMASS/amass_data/",
        amass_text_path="/nfs/USRCSEA/IVA/Datasets/HumanML3D/texts/",
        amass_text_mapping_file_all="/nfs/USRCSEA/IVA/Datasets/AMASS/index.csv",
        amass_split="all",
        interhuman_path="/nfs/USRCSEA/IVA/Datasets/InterHuman/motions/",
        interhuman_text_path="/nfs/USRCSEA/IVA/Datasets/InterHuman/annots/",
        webvid_path="/nfs/USRCSEA/IVA/Datasets/trace_motion/webvid_filter_03.txt",
        motion_use_dummy_betas=True,
        motion_use_dummy_hand_joints=True,
        motion_use_dummy_text=False,
        amass_trans_aug_rad=None,
        interhuman_trans_aug_rad=None,
        webvid_trans_aug_rad=None,
        # args for pose dataset
        pose_urls="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/yutao_text_rewrite/{00002..00700}.tar",
        pose_val_urls="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/yutao_text_rewrite/{00000..00001}.tar",
        pose_val=False,
        min_poses=1, # number of poses that each returned sample in the batch should have
        max_poses=10, # maximum number of poses that we keep
        pose_use_dummy_betas=True,
        pose_use_dummy_text=False,
        pose_trans_aug_rad=None,
        pose_shuffle=2000,
        concat_vars=True,  # must be True
        yield_cond_dict=True,  # must be True
        # shared mean and std
        normalize=False,
        mean_path=None,
        std_path=None,
    ):

        self.motion_max_len = motion_max_len
        self.max_poses = max_poses

        self.laion_pose_batch_size = int(data_split[0] * batch_size)
        self.amass_batch_size = int(data_split[1] * batch_size)
        self.interhuman_batch_size = int(data_split[2] * batch_size)
        self.webvid_motion_batch_size = int(data_split[3] * batch_size)
        self.comp_batch_size = int(data_split[4] * batch_size)

        assert self.laion_pose_batch_size > 0 or \
                self.amass_batch_size > 0 or \
                self.interhuman_batch_size > 0 or \
                self.webvid_motion_batch_size > 0 or \
                self.comp_batch_size > 0

        assert concat_vars and yield_cond_dict

        # initial dataloaders

        if self.laion_pose_batch_size > 0:
            self.laion_pose_loader = LaionPose(
                urls=pose_urls,
                min_poses=min_poses,
                max_poses=max_poses,
                use_dummy_betas=pose_use_dummy_betas,
                use_dummy_text=pose_use_dummy_text,
                val=pose_val,
                val_urls=pose_val_urls,
                batch_size=self.laion_pose_batch_size,
                resampled=shuffle,
                normalize=normalize,
                mean_path=mean_path,
                std_path=std_path,
                trans_aug_rad=pose_trans_aug_rad,
            )

        if self.amass_batch_size > 0:
            self.amass_dataset = AmassDataset(
                max_len=motion_max_len,
                motion_path=amass_path,
                text_path=amass_text_path,
                text_mapping_file_all=amass_text_mapping_file_all,
                split=amass_split,
                use_dummy_betas=motion_use_dummy_betas,
                use_dummy_hand_joints=motion_use_dummy_hand_joints,
                use_dummy_text=motion_use_dummy_text,
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
                motion_path=interhuman_path,
                text_path=interhuman_text_path,
                use_dummy_betas=motion_use_dummy_betas,
                use_dummy_hand_joints=motion_use_dummy_hand_joints,
                use_dummy_text=motion_use_dummy_text,
                normalize=normalize,
                mean_path=mean_path,
                std_path=std_path,
                trans_aug_rad=interhuman_trans_aug_rad,
            )
            self.interhuman_loader = make_dataloader(
                self.interhuman_dataset, batch_size=self.interhuman_batch_size, shuffle=shuffle
            )

        if self.webvid_motion_batch_size > 0:
            self.webvid_motion_dataset = WebVidMotionDataset(
                max_len=motion_max_len,
                max_pose=max_poses,
                motion_list_path=webvid_path,
                use_dummy_betas=motion_use_dummy_betas,
                use_dummy_hand_joints=motion_use_dummy_hand_joints,
                use_dummy_text=motion_use_dummy_text,
                normalize=normalize,
                mean_path=mean_path,
                std_path=std_path,
                trans_aug_rad=webvid_trans_aug_rad,
            )
            self.webvid_motion_loader = make_dataloader(
                self.webvid_motion_dataset, batch_size=self.webvid_motion_batch_size, shuffle=shuffle
            )

        if self.comp_batch_size > 0:
            self.comp_loader = AmassCompDataset(
                batch_size=self.comp_batch_size,
                max_poses=max_poses,
                shuffle=shuffle,
                max_len=motion_max_len,
                motion_path=amass_path,
                text_path=amass_text_path,
                text_mapping_file_all=amass_text_mapping_file_all,
                split=amass_split,
                use_dummy_betas=motion_use_dummy_betas,
                use_dummy_hand_joints=motion_use_dummy_hand_joints,
                #use_dummy_text=motion_use_dummy_text,
                use_dummy_text=True,  # for now, treat as unconditional
                normalize=normalize,
                mean_path=mean_path,
                std_path=std_path,
                trans_aug_rad=amass_trans_aug_rad,
            )

    def __iter__(self):
        return self

    def _process_laion_pose(self, pose_laion):
        pose_laion = pose_laion.unsqueeze(1)
        pose_laion = get_pose_pad_batch(
                        pose_laion,
                        lengths=[1] * pose_laion.shape[0],
                        num_poses=[self.max_poses] * pose_laion.shape[0],
                        max_len=self.motion_max_len,
                        max_poses=self.max_poses
                    )
        return pose_laion

    def _process_amass_pose(self, pose_amass):
        pose_amass = pose_amass.unsqueeze(2)
        pose_amass = get_pose_pad_batch(
                        pose_amass,
                        lengths=[self.motion_max_len] * pose_amass.shape[0],
                        num_poses=[1] * pose_amass.shape[0],
                        max_len=self.motion_max_len,
                        max_poses=self.max_poses
                    )
        return pose_amass

    def _process_interhuman_pose(self, pose_interhuman):
        pose_interhuman = get_pose_pad_batch(
                        pose_interhuman,
                        lengths=[self.motion_max_len] * pose_interhuman.shape[0],
                        num_poses=[2] * pose_interhuman.shape[0],
                        max_len=self.motion_max_len,
                        max_poses=self.max_poses
                    )
        return pose_interhuman

    def _process_webvid_motion_pose(self, pose_webvid_motion):
        return pose_webvid_motion

    def _process_comp_pose(self, pose_comp):
        return pose_comp

    def _merge_conditions(
        self, cond_dict_list, shared_keys=('text', 'num_poses', 'lengths',)
    ):

        # add 'num_poses' key to motion data and 'lengths' key to pose data
        for cond_dict in cond_dict_list:
            if 'num_poses' not in cond_dict:
                cond_dict['num_poses'] = torch.tensor([1] * len(cond_dict['lengths']))
            if 'lengths' not in cond_dict:
                cond_dict['lengths'] = torch.tensor([1] * len(cond_dict['num_poses']))

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
        if self.laion_pose_batch_size > 0:
            pose_laion, cond_dict_laion = next(self.laion_pose_loader)
            pose_laion = self._process_laion_pose(pose_laion)
            all_poses.append(pose_laion)
            all_cond_dict.append(cond_dict_laion)
        if self.amass_batch_size > 0:
            pose_amass, cond_dict_amass = next(self.amass_loader)
            pose_amass = self._process_amass_pose(pose_amass)
            all_poses.append(pose_amass)
            all_cond_dict.append(cond_dict_amass)
        if self.interhuman_batch_size > 0:
            pose_interhuman, cond_dict_interhuman = next(self.interhuman_loader)
            pose_interhuman = self._process_interhuman_pose(pose_interhuman)
            all_poses.append(pose_interhuman)
            all_cond_dict.append(cond_dict_interhuman)
        if self.webvid_motion_batch_size > 0:
            pose_webvid_motion, cond_dict_webvid_motion = next(self.webvid_motion_loader)
            pose_webvid_motion = self._process_webvid_motion_pose(pose_webvid_motion)
            all_poses.append(pose_webvid_motion)
            all_cond_dict.append(cond_dict_webvid_motion)
        if self.comp_batch_size > 0:
            pose_comp, cond_dict_comp = next(self.comp_loader)
            pose_comp = self._process_comp_pose(pose_comp)
            all_poses.append(pose_comp)
            all_cond_dict.append(cond_dict_comp)

        if len(all_poses) == 0:
            raise ValueError('Either amass batch size or laion pose batch size or interhuman batch size must be positive.')
        pose_out = torch.cat(all_poses, dim=0)
        # merge conditions into single dictionary
        cond_dict = self._merge_conditions(all_cond_dict)

        if pose_out.shape[1] == 1:
            # pose mode
            pose_out = pose_out.squeeze(1)
            del cond_dict['lengths']

        return pose_out, cond_dict


if __name__ == '__main__':

    import os
    from datetime import datetime

    from viz.viz_utils import viz_from_loader

    save_dir = 'out_joint_viz'
    save_dir = os.path.join(save_dir, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))

    batch_size = 30
    data_split = (0.5, 0.2, 0.2, 0.1, 0.0)
    frameskip = 10

    joint_loader = JointMotionPose(
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
        unsqueeze_pose_dim=False,
        unsqueeze_motion_dim=False,
    )
