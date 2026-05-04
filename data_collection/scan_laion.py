import os
import io
import argparse
import braceexpand
import tarfile

import torch
import torch.distributed as dist
from PIL import Image
import json

import time
from pdb import set_trace as st


def scan_laion(
    args,
    world_rank,
    device,
    world_size,
    base_tar_path="/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data2",
    expected_keys=(
        'NSFW',
        'similarity',
        'LICENSE',
        'caption',
        'url',
        'key',
        'status',
        'error_message',
        'width',
        'height',
        'original_width',
        'original_height',
        'exif',
        'sha256'
    )
):

    for i, tar_file in enumerate(args.tar_path):
        if i > 0:
            print(f'GPU {device} time for tarfile: {time.time() - tar_start_time}')
        print(f'GPU {device} scanning tarfile {tar_file}, number {i+1} of {len(args.tar_path)}')
        tar_start_time = time.time()
        tf = tarfile.open(os.path.join(base_tar_path, tar_file))
        for j, sub_file in enumerate(tf):
            if sub_file.name.endswith('jpg'):
                if i > 0 and not found_json:
                    print(f'Missing json file for image {last_jpg_stem}.jpg')
                if i > 0 and not found_txt:
                    print(f'Missing txt file for image {last_jpg_stem}.jpg')
                last_jpg_stem = sub_file.name.split('.')[0]
                found_json = False
                found_txt = False
            else:
                if not sub_file.name.split('.')[0] == last_jpg_stem:
                    print(f'Missing jpg file for {sub_file.name}')
                else:
                    if sub_file.name.endswith('json'):
                        found_json = True
                    if sub_file.name.endswith('txt'):
                        found_txt = True
            if sub_file.name.endswith('jpg'):
                tarinfo = tf.getmember(sub_file.name)
                image = tf.extractfile(tarinfo)
                image = image.read()
                try:
                    image = Image.open(io.BytesIO(image))
                    image.verify()
                except:
                    print(f'Bad image (GPU {device}): {sub_file.name}')
            if sub_file.name.endswith('json'):
                tarinfo = tf.getmember(sub_file.name)
                json_file = tf.extractfile(tarinfo)
                json_file = json_file.read()
                json_data = json.loads(json_file)
                if not tuple(json_data.keys()) == expected_keys:
                    print(f'Bad json with missing keys (GPU {device}): {sub_file.name}')
                    print(f'Found keys: {tuple(json_data.keys())}. Expected keys: {expected_keys}')
                if not 'caption' in json_data.keys() or not isinstance(json_data['caption'], str):
                    print(f'Bad json with missing or invalid caption (GPU {device}): {sub_file.name}')
        print(f'Tarfile {tar_file} Total Files: {j + 1}. Total Samples: {(j + 1) / 3}')
        if not (j + 1) % 3 == 0:
            print(f'Tarfile {tar_file} number of files not divisible by 3, incomplete file list.')
        tf.close()


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_path", type=str, default="{00000..41407}.tar")
    parser.add_argument(
        '--local_rank',
        type=int,
        help='Local rank for torch.distributed.launch function. (unused legacy argument)',
    )
    parser.add_argument('--MASTER', type=str, default='127.0.0.1', help='Master node address.')

    args = parser.parse_args()

    # add timestamp to output folder path
    os.environ['MASTER_ADDR'] = args.MASTER

    return args


if __name__ == "__main__":
    # rank info passed in through env
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    print('World Rank:   ', WORLD_RANK)
    print('Local Rank:   ', LOCAL_RANK)
    if WORLD_RANK == 0:
        print('World Size:   ', WORLD_SIZE)

    # args for running code
    args = setup_args()

    # dist setup
    torch.cuda.set_device(LOCAL_RANK)
    torch.cuda.empty_cache()
    dist.init_process_group('nccl', rank=WORLD_RANK, world_size=WORLD_SIZE)

    # list all paths and split across workers
    args.tar_path = list(braceexpand.braceexpand(args.tar_path))
    args.tar_path = args.tar_path[WORLD_RANK:][::WORLD_SIZE]

    # run data collection
    scan_laion(args, WORLD_RANK, LOCAL_RANK, WORLD_SIZE)
