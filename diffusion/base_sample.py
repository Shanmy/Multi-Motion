"""
Generate a large batch of samples from a model and save them as a large
numpy array. This can be used to produce samples for FID/FVD evaluation.
"""

import os
from omegaconf import OmegaConf

import numpy as np

import torch
import torch.distributed as dist

from diffusion.dist_util import (
    dist_print, 
    setup_exp,
    gather_sample_outputs,
    save_text_prompts,
)
from diffusion.script_util import setup_eval_model, create_gaussian_diffusion, create_data_loader
from evaluations.sample_diffusion import sample_diffusion
from evaluations.sample_diffusion_2_stage import sample_diffusion_2_stage
from evaluations.denoised_fn import get_cond_model_dict
from evaluations.fid import get_eval_metrics_from_saved_files

from viz.viz_utils import viz_from_loader


def main(config, world_rank, local_rank, world_size):
    args = OmegaConf.create(config)

    dist_print("creating model and diffusion...")

    model = setup_eval_model(
        model_type=args.model_type,
        model_args=args.model,
        local_rank=local_rank,
        checkpoint=args.checkpoint if 'checkpoint' in args.keys() else None,
        guidance_param=args.guidance_param if 'guidance_param' in args.keys() else None,
    )

    # load up to three types of possible helper models: pose, uncond motion, cond motion
    if 'cond_model_args' in args.keys() and args.cond_model_args is not None:
        assert args.diffusion.model_mean_type == 'xstart', "Conditioning models currently must predict xstart."
        cond_model_dict = get_cond_model_dict(args.cond_model_args, local_rank)
    else:
        cond_model_dict = None

    diffusion = create_gaussian_diffusion(**args.diffusion)

    dist_print("creating data loader...")
    val_data = create_data_loader(args.dataset, batch_size=args.batch_size, val=True)

    dist_print("sampling...")

    use_2_stage_sampler = (
        args.use_2_stage_sampler if 'use_2_stage_sampler' in args.keys() and args.use_2_stage_sampler
        else False
    )
    if not 'model_type_1st_stage' in args.keys():
        sample, cond_dict = sample_diffusion(
            model=model,
            diffusion=diffusion,
            num_samples=int(np.ceil(args.eval_num_samples / world_size)),
            local_rank=local_rank,
            dataloader=val_data,  # currently only support samples using conds from dataloader. to use new prompts, make dummy dataloader.
            mean_path=None,
            std_path=None,
            num_sample_frames=(args.num_sample_frames if 'num_sample_frames' in args.keys() else None),
            cond_model_dict=cond_model_dict,
        )
    else:
        model_1st_stage = setup_eval_model(
            model_type=args.model_type_1st_stage,
            model_args=args.model_1st_stage,
            local_rank=local_rank,
            checkpoint=args.checkpoint_1st_stage,
            guidance_param=args.guidance_param_1st_stage,
        )
        sample, cond_dict = sample_diffusion_2_stage(
            model_1st_stage=model_1st_stage,
            model_2nd_stage=model,
            diffusion=diffusion,
            num_samples=int(np.ceil(args.eval_num_samples / world_size)),
            local_rank=local_rank,
            dataloader=val_data,  # currently only support samples using conds from dataloader. to use new prompts, make dummy dataloader.
            mean_path=None,
            std_path=None,
            num_sample_frames=(args.num_sample_frames if 'num_sample_frames' in args.keys() else None),
            cond_model_dict=cond_model_dict,
        )

    # gather results across gpus
    sample_all, text_all, num_frames_all, num_poses_all = gather_sample_outputs(sample, cond_dict, local_rank)
    if args.pose_type == 'multipose':
        sample_all = sample_all.unsqueeze(1)
        num_frames_all = torch.ones_like(num_poses_all)
    if args.pose_type == 'motion':
        sample_all = sample_all.unsqueeze(2)
        num_poses_all = torch.ones_like(num_frames_all)

    if dist.get_rank() == 0:

        out_path_stem = os.path.join(args.out_dir, f"samples_{'x'.join([str(x) for x in sample_all.shape])}")
        sample_save_file = out_path_stem + '.npz'
        text_save_file = out_path_stem + '.txt'
        dist_print(f"saving samples to {sample_save_file}")
        save_dict = {"sample": sample_all.numpy()}
        if num_frames_all is not None:
            save_dict["num_frames"] = num_frames_all.numpy()
        if num_poses_all is not None:
            save_dict["num_poses"] = num_poses_all.numpy()
        np.savez(sample_save_file, **save_dict)
        save_text_prompts(text_save_file, text_all)

        if args.get_eval_metrics:
            get_eval_metrics_from_saved_files(
                sample_save_file,
                text_save_file,
                gt_pose_path=(
                    args.eval_gt_pose_file if 'eval_gt_pose_file' in args.keys() else None
                ),
                gt_pose_text_path=(
                    args.eval_gt_pose_text_file if 'eval_gt_pose_text_file' in args.keys() else None
                ),
                gt_motion_path=(
                    args.eval_gt_motion_file if 'eval_gt_motion_file' in args.keys() else None
                ),
                gt_motion_text_path=(
                    args.eval_gt_motion_text_file if 'eval_gt_motion_text_file' in args.keys() else None
                ),
            )

        if args.viz:
            viz_from_loader(
                poses=sample[0:args.viz_num_samples],
                cond_dict={
                    k: v[0:args.viz_num_samples] for k, v in cond_dict.items()
                },
                save_dir=args.out_dir,
                frameskip=(args.frameskip if 'frameskip' in args.keys() else 1),
                device=sample.device,
                mean_path=args.dataset.mean_path,
                std_path=args.dataset.std_path,
                unsqueeze_pose_dim=(args.pose_type == 'motion'),
                unsqueeze_motion_dim=(args.pose_type == 'multipose'),
            )

    dist.barrier()
    dist_print("sampling complete")


if __name__ == "__main__":
    config, world_rank, local_rank, world_size = setup_exp(__file__)
    main(config, world_rank, local_rank, world_size)
