import braceexpand

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import webdataset as wds

from clip_tokenizer import tokenize as clip_tokenizer
from data_utils import (
    get_pose_from_smpl_params,
    make_shorter_instruct_blip_text,
    get_pose_pad,
    pose_random_rotation,
)


class LaionPoseWDS:
    def __init__(
        self,

        # tarfiles 00000 and 00001 for val, tarfiles beyond 701 for training clip pose model. remaining files for training.
        urls="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/yutao_text_rewrite/{00002..00700}.tar",
        val_urls="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/yutao_text_rewrite/{00000..00001}.tar",
        val=False,

        # original data below (DO NOT USE)
        # tarfiles 00000 and 00001 for val, 715 to 815 for training clip pose model. remaining files for training.
        #urls="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/v2/{00002..00714}.tar",
        #val_urls="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/v2/{00000..00001}.tar",

        min_poses=1, # number of poses that each returned sample in the batch should have
        max_poses=10, # maximum number of poses that we keep
        use_dummy_betas=True,  # whether or not to overwrite shape param betas with neutral zero vector
        use_dummy_text=False,
        normalize=False,
        mean_path=None,
        std_path=None,
        process_trans=True,
        randomly_rotate_pose=True,
        trans_aug_rad=None,
        align_pose=False,  # needs to be True for /v2/ version of data, False for /yutao_text_rewrite/ version
        tokenizer=None,
        shorten_instruct_blip_text=True,
    ):

        self.min_poses = min_poses
        self.max_poses = max_poses
        self.use_dummy_betas = use_dummy_betas
        self.use_dummy_text = use_dummy_text
        self.normalize = normalize
        self.process_trans = process_trans
        self.randomly_rotate_pose = randomly_rotate_pose
        self.trans_aug_rad = trans_aug_rad
        self.align_pose = align_pose
        self.shorten_instruct_blip_text = shorten_instruct_blip_text

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

        if not val:
            # training samples
            self.urls = list(braceexpand.braceexpand(urls))
        else:
            # load held-out validation samples
            self.urls = list(braceexpand.braceexpand(val_urls))
        if tokenizer is None:
            tokenizer = clip_tokenizer
        self.tokenizer = tokenizer

        def pad_pose_sample(sample):
            poses_pad = get_pose_pad(sample, max_len=self.max_poses, return_npy=True)
            return poses_pad
        self.pad_pose_sample = pad_pose_sample

        def prepare_pose_sample(sample_dict):
            if self.use_dummy_betas:
                # set betas to 0 to match amass/interhuman
                sample_dict["beta"] = np.zeros_like(sample_dict["beta"])
            sample_dict["pose"], sample_dict["trans_shift"] = get_pose_from_smpl_params(
                                                                betas=sample_dict["beta"],
                                                                thetas=sample_dict["theta"],
                                                                trans=sample_dict["trans"],
                                                                process_trans=self.process_trans,
                                                                trans_aug_rad=self.trans_aug_rad,
                                                                convert_6d=self.align_pose,
                                                                align_pose=self.align_pose,
                                                                return_shift=True,
                                                            )
            if self.randomly_rotate_pose:
                sample_dict["pose"] = pose_random_rotation(sample_dict["pose"])
            if self.normalize:
                sample_dict["pose"] = (sample_dict["pose"] - self.mean) / self.std
            return sample_dict
        self.prepare_pose_sample = prepare_pose_sample

        def add_num_poses_key(dict_in, key="num_poses"):
            num_poses = dict_in["theta"].shape[0]
            dict_in[key] = torch.tensor(num_poses)  # better to have this as a torch tensor for joint loader
            return dict_in
        self.add_num_poses_key = add_num_poses_key

        # function to help us return samples with a specified number of poses
        # if the number of poses is >= num_poses_in_sample then keep it
        def filter_num_poses(dict_in):
            if dict_in['num_poses'] >= self.min_poses and dict_in['num_poses'] <= self.max_poses:
                return True
            else:
                return False
        self.filter_num_poses = filter_num_poses

        def preprocess_text(sample_dict):
            text_dict = sample_dict["text"]
            for k in text_dict.keys():
                sample_dict[k] = text_dict[k].lower()
            return sample_dict
        self.preprocess_text = preprocess_text

        def tokenize_text(text):
            # coin flip to determine if text comes from blip or laion captions
            if np.random.rand() < 0.33:
                text = text["laion_text"]
            else:
                if np.random.rand() < 0.5:
                    text = text["blip_text"]
                else:
                    text = text["instruct_blip_text"]
            text = text.lower()
            text_tokens = self.tokenizer(text)[0]
            return text_tokens
        self.tokenize_text = tokenize_text

        def prepare_cond_output(dict_in):
            cond_dict = {}
            cond_dict['num_poses'] = dict_in['num_poses']
            cond_dict['key'] = dict_in['key']

            if not self.use_dummy_text:
                # TODO: randomly select text here?
                instruct_blip_text = dict_in['text']['instruct_blip_text'].lower()
                if self.shorten_instruct_blip_text:
                    cond_dict['text'] = make_shorter_instruct_blip_text(instruct_blip_text)
                else:
                    cond_dict['text'] = instruct_blip_text
            else:
                # empty string for unconditional text
                cond_dict['text'] = ''

            # add cond dict to the sample dict
            dict_in["cond_dict"] = cond_dict
            return dict_in
        self.prepare_cond_output = prepare_cond_output

    def wds_loader(
        self,
        batch_size,
        num_workers=8,
        resampled=True,
        shuffle=2000,
        concat_vars=True,
        yield_cond_dict=True,
        # use options below only when yield_cond_dict is False
        yield_tokenized_text=False,
        yield_num_poses=False,
        yield_trans_shift=False,
        yield_key=False,
    ):
        dataset = wds.WebDataset(
            self.urls,
            resampled=resampled,
            handler=wds.ignore_and_continue,
            nodesplitter=nodesplitter,
        )
        if shuffle > 0:
            dataset = dataset.shuffle(shuffle)

        # steps to prepare pose data
        dataset = dataset.decode()
        dataset = dataset.rename(
            beta="beta.npy",
            theta="theta.npy",
            trans="trans.npy",
            text="text.json",
            key="__key__"
        )
        dataset = dataset.map(self.add_num_poses_key) # add in key for num_poses
        dataset = dataset.select(self.filter_num_poses) # only return samples with greater than specified number of poses

        if concat_vars:
            dataset = dataset.map(self.prepare_pose_sample)
            dataset = dataset.map_dict(pose=self.pad_pose_sample)
            yield_args = ("pose",)
        else:
            dataset = dataset.map_dict(
                beta=self.pad_pose_sample,
                theta=self.pad_pose_sample,
                trans=self.pad_pose_sample,
            )
            yield_args = ("beta", "theta", "trans")

        if yield_cond_dict:
            dataset = dataset.map(self.prepare_cond_output)
            yield_args += ("cond_dict",)
        else:
            if yield_tokenized_text:
                dataset.map_dict(text=self.tokenize_text)
                yield_args += ("text",)
            else:
                dataset = dataset.map(self.preprocess_text)
                yield_args += ("laion_text", "blip_text", "instruct_blip_text",)
            if yield_num_poses:
                yield_args += ("num_poses",)
            if yield_trans_shift:
                dataset = dataset.map_dict(trans_shift=self.pad_pose_sample)
                yield_args += ("trans_shift",)
            if yield_key:
                yield_args += ("key",)

        dataset = dataset.to_tuple(*yield_args).batched(batch_size, partial=False)
        data_loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers)
        return data_loader


def nodesplitter(src, group=None):
    if dist.is_initialized():
        if group is None:
            group = dist.group.WORLD
        rank = dist.get_rank(group=group)
        size = dist.get_world_size(group=group)
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src


# wrapper to reformat condition dict output from WDS
class LaionPose:
    def __init__(
        self,
        # args for setting up LaionPoseWDS
        # tarfiles 00000 and 00001 for val, tarfiles beyond 701 for training clip pose model. remaining files for training.
        urls="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/yutao_text_rewrite/{00002..00700}.tar",
        val_urls="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/yutao_text_rewrite/{00000..00001}.tar",
        val=False,

        # original data below (DO NOT USE)
        # tarfiles 00000 and 00001 for val, 715 to 815 for training clip pose model. remaining files for training.
        #urls="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/v2/{00002..00714}.tar",
        #val_urls="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/v2/{00000..00001}.tar",

        min_poses=1, # number of poses that each returned sample in the batch should have
        max_poses=10, # maximum number of poses that we keep
        use_dummy_betas=True,
        use_dummy_text=False,
        normalize=False,
        mean_path=None,
        std_path=None,
        process_trans=True,
        randomly_rotate_pose=True,
        trans_aug_rad=None,
        align_pose=False,  # needs to be True for /v2/ version of data, False for /yutao_text_rewrite/ version
        # args for dataloader
        batch_size=1,
        num_workers=8,
        resampled=True,
        shuffle=2000,
        concat_vars=True,  # must be True
        yield_cond_dict=True,  # must be True, otherwise just use LaionPoseWDS dataset
    ):
        self.max_poses = max_poses
        self.laion_pose = LaionPoseWDS(
            urls=urls,
            min_poses=min_poses,
            max_poses=max_poses,
            use_dummy_betas=use_dummy_betas,
            use_dummy_text=use_dummy_text,
            val=val,
            val_urls=val_urls,
            normalize=normalize,
            mean_path=mean_path,
            std_path=std_path,
            process_trans=process_trans,
            randomly_rotate_pose=randomly_rotate_pose,
            trans_aug_rad=trans_aug_rad,
            align_pose=align_pose,
            tokenizer=None,
        )
        assert concat_vars, 'Must concatenate all pose vars to use this loader.'
        assert yield_cond_dict, 'Must set "yield_cond_dict" to True, otherwise should use LaionPoseWDS dataset.'
        self.laion_dataloader = self.laion_pose.wds_loader(
            batch_size=batch_size,
            num_workers=num_workers,
            resampled=resampled,
            shuffle=shuffle,
            concat_vars=concat_vars,
            yield_cond_dict=yield_cond_dict,
            yield_tokenized_text=False,
            yield_key=False,
        )
        self.laion_iterator = iter(self.laion_dataloader)

    def __iter__(self):
        # create new iterator with different random order
        self.laion_iterator = iter(self.laion_dataloader)
        return self

    def __next__(self):
        pose, cond_dict_orig = next(self.laion_iterator)
        # repack cond dict so each dict key gives info across entire batch
        cond_dict ={
            k: (
                torch.stack([dict_entry[k] for dict_entry in cond_dict_orig]) if torch.is_tensor(cond_dict_orig[0][k])
                else [dict_entry[k] for dict_entry in cond_dict_orig]
            )
            for k in cond_dict_orig[0].keys() 
        }
        return pose, cond_dict


if __name__ == "__main__":

    import os
    from datetime import datetime
    from torch.utils.data import DataLoader

    from viz.viz_utils import viz_from_loader

    save_dir = 'out_laion_viz'
    save_dir = os.path.join(save_dir, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    batch_size = 10

    pose_dataloader = LaionPose(
        normalize=False,
        batch_size=batch_size,
        use_dummy_betas=True,
        resampled=False,
        shuffle=0,
        num_workers=1,
        randomly_rotate_pose=True,
        concat_vars=True,
        yield_cond_dict=True,
    )

    pose_sample = next(iter(pose_dataloader))

    viz_from_loader(
        poses=pose_sample[0],
        cond_dict=pose_sample[1],
        save_dir=save_dir,
        mean_path=None,
        std_path=None,
        unsqueeze_pose_dim=False,
        unsqueeze_motion_dim=True,
    )
