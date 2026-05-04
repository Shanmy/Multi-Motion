import braceexpand

import numpy as np

import torch.distributed as dist
from torch.utils.data import DataLoader

import webdataset as wds

import sys
sys.path.append('../')

from clip_tokenizer import tokenize as tokenizer_clip


class LaionPose:
    def __init__(
        self,
        urls="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/v1/{00000..01596}.tar",
        tokenizer=None,
    ):

        self.urls = list(braceexpand.braceexpand(urls))
        if tokenizer is None:
            tokenizer = tokenizer_clip
        self.tokenizer = tokenizer

        def preprocess_npy(npy_data, max_num_poses=10):
            num_poses = npy_data.shape[0]
            if num_poses < max_num_poses:
                # pad empty poses with zeros
                pad_poses = max_num_poses - num_poses
                npy_data = np.concatenate((npy_data, np.zeros(shape=[pad_poses, npy_data.shape[1]])), 0)
            else:
                # select max number of poses
                npy_data = npy_data[:max_num_poses]
            return npy_data.astype(np.float32)
        self.preprocess_npy = preprocess_npy

        def concat_theta_and_trans(sample_dict):
            sample_dict["theta_trans"] = np.concatenate((sample_dict["theta"], sample_dict["trans"]), 1)
            return sample_dict
        self.concat_theta_and_trans = concat_theta_and_trans

        def preprocess_text(sample_dict):
            text_dict = sample_dict["text"]
            sample_dict["blip_text"] = text_dict["blip_text"][0]
            sample_dict["laion_text"] = text_dict["laion_text"][0]
            return sample_dict
        self.preprocess_text = preprocess_text

        def tokenize_text(text, blip_prob=1.0):
            # coin flip to determine if text comes from blip or laion captions
            if np.random.rand() < blip_prob:
                text = text["blip_text"][0]
            else:
                text = text["laion_text"][0]
            text_tokens = self.tokenizer(text)[0]
            return text_tokens
        self.tokenize_text = tokenize_text

    def wds_loader(
        self,
        batch_size,
        num_workers=8,
        resampled=False,
        shuffle=2000,
        concat_theta_and_trans=False,
        yield_tokenized_text=False,
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
        dataset = dataset.rename(theta="theta.npy", trans="trans.npy", text="text.json", key="__key__")
        dataset = dataset.map_dict(theta=self.preprocess_npy, trans=self.preprocess_npy)

        yield_args = ()

        if concat_theta_and_trans:
            # "theta" and "trans" get concatenated into new data arg "theta_trans"
            dataset = dataset.map(self.concat_theta_and_trans)
            yield_args += ("theta_trans",)
        else:
            # separate theta and trans (good for single person experiments)
            yield_args += ("theta", "trans")
        # TODO: add option to yield theta for only one person?

        if yield_tokenized_text:
            dataset.map_dict(text=self.tokenize_text)
            yield_args += ("text",)
        else:
            dataset = dataset.map(self.preprocess_text)
            yield_args += ("blip_text", "laion_text")

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
