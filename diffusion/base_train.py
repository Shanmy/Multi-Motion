"""
Train a diffusion model on pose or motion data.
"""

import os
from omegaconf import OmegaConf

import torch

from diffusion.dist_util import dist_print, setup_exp
from diffusion.resample import create_named_schedule_sampler
from diffusion.script_util import create_model, create_gaussian_diffusion, create_data_loader
from diffusion.train_util import TrainLoop

from viz.viz_utils import viz_from_loader


def main(config, world_rank, local_rank, world_size):
    args = OmegaConf.create(config)

    dist_print("creating model and diffusion...")
    model = create_model(args.model_type, **args.model)
    model.to(local_rank)
    if 'model_type_1st_stage' in args.keys() and args.model_type_1st_stage is not None:
        # 1st stage model for sampling eval during training
        model_1st_stage = create_model(args.model_type_1st_stage, **args.model_1st_stage)
        ckpt_1st = args.get('checkpoint_1st_stage', None)
        if ckpt_1st:
            model_1st_stage.load_state_dict(torch.load(ckpt_1st, map_location="cpu"), strict=False)
        else:
            dist_print("WARNING: checkpoint_1st_stage not set; 1st-stage eval model has random weights.")
        model_1st_stage.to(local_rank)
        model_1st_stage.eval()
    else:
        model_1st_stage = None
    diffusion = create_gaussian_diffusion(**args.diffusion)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    dist_print("creating data loader...")
    data = create_data_loader(args.dataset, batch_size=args.batch_size)
    eval_data_args = (
        args.eval_dataset if 'eval_dataset' in args.keys()
        else args.dataset
    )
    val_data = create_data_loader(eval_data_args, batch_size=args.eval_batch_size, val=True)

    if args.viz_gt and world_rank == 0:
        sample_folder = os.path.join(args.out_dir, 'viz', 'data')
        unsqueeze_pose_dim = (args.eval_pose_type == 'motion')
        unsqueeze_motion_dim = (args.eval_pose_type == 'multipose')
        pose_sample = next(data)
        viz_from_loader(
            poses=pose_sample[0],
            cond_dict=pose_sample[1],
            device=local_rank,
            save_dir=sample_folder,
            mean_path=args.dataset.mean_path,
            std_path=args.dataset.std_path,
            unsqueeze_pose_dim=unsqueeze_pose_dim,
            unsqueeze_motion_dim=unsqueeze_motion_dim,
            max_viz=args.eval_num_samples,
            frameskip=args.viz_frameskip,
        )

    dist_print("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        val_data=val_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        out_dir=args.out_dir,
        local_rank=local_rank,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        eval_num_samples=args.eval_num_samples,
        eval_interval=args.eval_interval,
        eval_batch_size=args.eval_batch_size,
        get_eval_metrics=args.get_eval_metrics,
        eval_max_viz_samples=args.eval_max_viz_samples,
        eval_gt_pose_file=(
            args.eval_gt_pose_file if 'eval_gt_pose_file' in args.keys() else None
        ),
        eval_gt_pose_text_file=(
            args.eval_gt_pose_text_file if 'eval_gt_pose_text_file' in args.keys() else None
        ),
        eval_gt_motion_file=(
            args.eval_gt_motion_file if 'eval_gt_motion_file' in args.keys() else None
        ),
        eval_gt_motion_text_file=(
            args.eval_gt_motion_text_file if 'eval_gt_motion_text_file' in args.keys() else None
        ),
        model_1st_stage=model_1st_stage,
        eval_num_sample_frames=(
            args.eval_num_sample_frames if 'eval_num_sample_frames' in args.keys() else None
        ),
        eval_pose_type=args.eval_pose_type,
        guidance_param=args.guidance_param,
        mean_path=args.dataset.mean_path,
        std_path=args.dataset.std_path,
        viz_frameskip=args.viz_frameskip,
        # unused fp 16 args
        use_fp16=False,
        fp16_scale_growth=None,
    ).run_loop()


if __name__ == "__main__":
    config, world_rank, local_rank, world_size = setup_exp(__file__)
    main(config, world_rank, local_rank, world_size)
