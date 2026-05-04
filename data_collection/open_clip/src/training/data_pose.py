import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler

import webdataset as wds
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

import sys
sys.path.append('../../../pose_datasets')
from amass_loader import AmassDataset
from data_utils import (
    get_pose_from_smpl_params,
    pose_random_rotation,
    make_shorter_instruct_blip_text,
    make_dataloader
)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, is_train=True, epoch=0, floor=False, tokenizer=None):

    def pad_pose_sample(sample, dummy_val=0, max_poses=10):
        num_poses = sample.shape[0]
        if num_poses < max_poses:
            # pad empty poses with zeros
            num_pad_poses = max_poses - num_poses
            dummy_poses = dummy_val * np.ones(shape=[num_pad_poses, sample.shape[1]])
            sample = np.concatenate((sample, dummy_poses), axis=0)
        else:
            # select max number of poses
            sample = sample[:max_poses]
        return sample.astype(np.float32)

    def prepare_pose_sample(sample_dict, use_dummy_betas=True, process_trans=True, trans_aug_rad=None, align_pose=False):

        if use_dummy_betas:
            # set betas to 0 to match amass/interhuman
            sample_dict["beta"] = np.zeros_like(sample_dict["beta"])
        sample_dict["pose"] = get_pose_from_smpl_params(
                                        betas=sample_dict["beta"],
                                        thetas=sample_dict["theta"],
                                        trans=sample_dict["trans"],
                                        process_trans=process_trans,
                                        trans_aug_rad=trans_aug_rad,
                                        convert_6d=align_pose,
                                        align_pose=align_pose,
                                        return_shift=False,
                                    )
        sample_dict["pose"] = pose_random_rotation(sample_dict["pose"])
        return sample_dict

    def tokenize_text(text):
        # coin flip to determine if text comes from blip or laion captions
        #if np.random.rand() < 0.33:
        #    text = text["laion_text"]
        #else:
        #    if np.random.rand() < 0.5:
        #        text = text["blip_text"]
        #    else:
        #        text = text["instruct_blip_text"]

        # just use instruct blip caption :)
        text = text["instruct_blip_text"].lower()
        text = make_shorter_instruct_blip_text(text)

        text_tokens = tokenizer(text)[0]
        return text_tokens

    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        assert args.train_data_upsampling_factors is None,\
            "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.decode(),
        wds.rename(
            beta="beta.npy",
            theta="theta.npy",
            trans="trans.npy",
            text="text.json"
        ),
        wds.map(prepare_pose_sample),
        wds.map_dict(pose=pad_pose_sample, text=tokenize_text),
        wds.to_tuple("pose", "text"),
        wds.batched(args.batch_size, partial=not is_train),
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


class AmassDatasetLoader:
    def __init__(self, dataloader, batch_size, num_samples, tokenizer=None):
        self.dataloader = dataloader
        self.num_samples = num_samples
        self.num_batches = num_samples // batch_size
        self.tokenizer = tokenizer

    def __iter__(self):
        self.data_iterator = iter(self.dataloader)
        return self

    def __next__(self):
        pose, cond = next(self.data_iterator)
        text_tokens = self.tokenizer(cond["text"])
        return pose, text_tokens


class AmassDatasetCLMP:
    def __init__(self, dataloader, batch_size, num_samples, tokenizer=None):
        self.dataloader = AmassDatasetLoader(dataloader, batch_size, num_samples, tokenizer=tokenizer)

    def set_epoch(self, epoch):
        pass


def get_amass_data(batch_size, tokenizer=None):
    amass_dataset = AmassDataset(
                        max_len=61,
                        motion_path="/nfs/USRCSEA/IVA/Datasets/AMASS/amass_data/",
                        text_path="/nfs/USRCSEA/IVA/Datasets/HumanML3D/texts/",
                        split="train",
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
                    )
    amass_loader = DataLoader(amass_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    amass_data = AmassDatasetCLMP(amass_loader, batch_size, len(amass_dataset), tokenizer=tokenizer)
    return amass_data


def get_data_pose(args, epoch=0, tokenizer=None):

    if args.dataset_type == 'webdataset':
        data = {"train": get_wds_dataset(args, is_train=True, epoch=epoch, tokenizer=tokenizer)}
    elif args.dataset_type == 'amass':
        data = {"train": get_amass_data(args.batch_size, tokenizer=tokenizer)}
    else:
        raise ValueError('Invalid "data_type".')

    return data
