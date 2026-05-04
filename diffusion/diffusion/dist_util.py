"""
Helpers for distributed training.
"""

import os
import argparse
import importlib
import tempfile
from datetime import datetime, timedelta
from pkgutil import iter_modules

import torch
import torch.distributed as dist

import numpy as np


def sync_params(params):
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)


def gather_on_gpu(tensor):
    tensor_list_gather = [torch.zeros(tensor.shape, device=tensor.device)
                          for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list_gather, tensor)
    return tensor_list_gather


def gather_on_cpu(tensor, local_rank=None, batch_size=16):
    # input tensor should be on local device
    tensor_list_full = [torch.zeros([0] + list(tensor.shape[1:])) for _ in range(dist.get_world_size())]
    for i in range(int(np.ceil(tensor.shape[0] / batch_size))):
        start_ind = i * batch_size
        end_ind = min((i + 1) * batch_size, tensor.shape[0])
        tensor_batch = tensor[start_ind:end_ind]
        if local_rank is not None:
            tensor_batch = tensor_batch.to(local_rank)
        tensor_list_gather = gather_on_gpu(tensor_batch)
        tensor_list_full = [torch.cat((tensor_list_full[j], tensor_list_gather[j].cpu()), 0)
                            for j in range(len(tensor_list_full))]
    tensor_gather = torch.cat(tensor_list_full, 0)
    return tensor_gather


# gather all samples across gpus (for saving or running eval on the master worker)
def gather_sample_outputs(pose, cond, local_rank):
    pose_all = gather_on_cpu(pose, local_rank=local_rank)
    text_prompts_all = gather_text_prompts_on_master(cond['text'])
    if 'lengths' in cond.keys():
        if type(cond['lengths']) is list:
            cond['lengths'] = torch.stack(cond['lengths'])
        num_frames_all = gather_on_cpu(cond['lengths'].float(), local_rank=local_rank)
        num_frames_all = num_frames_all.long()
    else:
        num_frames_all = None
    if 'num_poses' in cond.keys():
        if type(cond['num_poses']) is list:
            cond['num_poses'] = torch.stack(cond['num_poses'])
        num_poses_all = gather_on_cpu(cond['num_poses'].float(), local_rank=local_rank)
        num_poses_all = num_poses_all.long()
    else:
        num_poses_all = None
    return pose_all, text_prompts_all, num_frames_all, num_poses_all


def dist_print(text, save_dir=None):
    if dist.get_rank() == 0:
        if save_dir:
            print(text, file=open(save_dir, 'a'))
        else:
            print(text)


def _save_file(file_in_name, file_out_name, exp_dir):
    file_in = open(file_in_name, 'r')
    file_out = open(os.path.join(exp_dir, 'code', file_out_name), 'w')
    for line in file_in:
        file_out.write(line)


def _save_code(exp_dir, package_name):

    modules = [[path, name, is_folder] for path, name, is_folder in iter_modules([package_name])]
    abs_path_chars = len(modules[0][0].path) - len(os.path.basename(modules[0][0].path))

    while not modules == []:
        modules_old = modules
        modules_new = []
        for path, module, is_folder in modules_old:
            if not is_folder:
                os.makedirs(os.path.join(exp_dir, 'code', path.path[abs_path_chars:]), exist_ok=True)
                # full path for in file
                file_in_name = os.path.join(path.path, module + '.py')
                # relative path for out file
                file_out_name = os.path.join(path.path[abs_path_chars:], module + '.py')
                # save a single file
                _save_file(file_in_name, file_out_name, exp_dir)
            else:
                modules_new.append(path.path[abs_path_chars:] + '/' + module)
            modules = [[path, name, is_folder] for path, name, is_folder in iter_modules(modules_new)]


def _setup(
    exp_dir,
    exec_files,
    save_folder_list=('diffusion', 'models', 'evaluations', '../pose_datasets', '../smpl', '../viz'),
    out_folder_list=('checkpoints', 'viz', 'log', 'code'),
):
    # make directory for saving results
    os.makedirs(exp_dir, exist_ok=True)
    for folder in list(out_folder_list):
        os.makedirs(os.path.join(exp_dir, folder), exist_ok=True)

    # save invidual exec files (python exec file, config file)
    for file_name in exec_files:
        _save_file(file_name, os.path.join(os.path.basename(file_name)), exp_dir)

    # save folders used for training
    for package_name in save_folder_list:
        # save all modules in experiment package
        _save_code(exp_dir, package_name)


def setup_eval_out_dir(checkpoint, world_rank):
    # default option to save samples in same folder as exp checkpoint
    out_dir = os.path.join(os.path.dirname(os.path.dirname(checkpoint)), 'samples')
    return out_dir


def setup_exp(exec_file_name, timeout_minutes=60):
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    print('World Rank:   ', WORLD_RANK)
    if WORLD_RANK == 0:
        print('World Size:   ', WORLD_SIZE)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file', 
        default='configs/default_config.py', 
        help='Name of config file.'
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        help='Local rank for torch.distributed.launch function. (unused legacy argument)'
    )
    parser.add_argument('--MASTER', type=str, default='127.0.0.1', help='Master node address.')
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.MASTER

    # get experiment config
    config_name = args.config_file.split('.')[0]
    config_module = importlib.import_module(config_name.replace('/', '.'))
    config = config_module.config

    if config['out_dir'] is None:
        # output directory inside of existing exp folder (eval only)
        config['out_dir'] = setup_eval_out_dir(config['checkpoint'], WORLD_RANK)
    # give exp_name unique timestamp identifier
    time_str = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    exp_name = os.path.basename(config_name) + '_' + str(WORLD_SIZE) + '_gpu'
    config['out_dir'] = os.path.join(config['out_dir'], exp_name, time_str)

    # setup dist
    torch.cuda.set_device(LOCAL_RANK)
    torch.cuda.empty_cache()
    dist.init_process_group(
        'nccl',
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
        timeout=timedelta(minutes=timeout_minutes)
    )

    # set up output folder and save exp code
    if WORLD_RANK == 0:
        _setup(config['out_dir'], [args.config_file, exec_file_name])

    return config, WORLD_RANK, LOCAL_RANK, WORLD_SIZE


def read_text_prompts(text_prompts_file):
    with open(text_prompts_file, 'r') as f:
        text_prompts_all = f.readlines()
    return text_prompts_all


def save_text_prompts(text_prompts_file, text_prompts, write_mode='w'):
    with open(text_prompts_file, write_mode) as f:
        for text_prompt in text_prompts:
            # remove new line characters which might split caption across lines
            text_prompt = text_prompt.replace('\n', "")
            # add line break at end of line to save each caption in separate line
            text_prompt += '\n'
            f.write(text_prompt)


def save_text_prompts_across_gpus(text_prompts_file, text_prompts):
    for gpu_id in range(dist.get_world_size()):
        if dist.get_rank() == gpu_id:
            save_text_prompts(text_prompts_file, text_prompts, write_mode='a')
        dist.barrier()


def gather_text_prompts_on_master(text_prompts):
    # Derive a run-unique name from the distributed env so concurrent jobs don't collide.
    run_tag = f"{os.environ.get('MASTER_ADDR', 'local')}_{os.environ.get('MASTER_PORT', '0')}".replace('.', '_')
    gather_file = os.path.join(tempfile.gettempdir(), f'gather_{run_tag}.txt')
    if dist.get_rank() == 0:
        if os.path.isfile(gather_file):
            os.remove(gather_file)
    dist.barrier()

    save_text_prompts_across_gpus(text_prompts_file=gather_file, text_prompts=text_prompts)

    if dist.get_rank() == 0:
        text_prompts_all = read_text_prompts(gather_file)
        os.remove(gather_file)
    else:
        text_prompts_all = None

    return text_prompts_all
