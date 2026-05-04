import torch

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

from models.pose_model import PoseNet
from models.joint_model import JointNet
from models.joint_model_2_stage import JointNet2Stage
from models.pose2motion import Pose2Motion
from models.cfg_sampler import ClassifierFreeSampleModel

from pose_datasets.amass_loader import AmassDataset
from pose_datasets.interhuman_loader import InterHumanDataset
from pose_datasets.laion_pose_loader import LaionPose
from pose_datasets.joint_loader import JointMotionPose
from pose_datasets.joint_uncond_motion_loader import JointMotionUncond
from pose_datasets.val_prompt_loader import ValPromptDataset
from pose_datasets.data_utils import make_dataloader


def create_data_loader(args, batch_size, val=False):
    if args.data_type == 'amass':
        dataset = AmassDataset(
            max_len=args.max_len,
            normalize=args.normalize,
            mean_path=(args.mean_path if args.normalize else None),
            std_path=(args.std_path if args.normalize else None),
            trans_aug_rad=(args.trans_aug_rad if 'trans_aug_rad' in args.keys() else None),
            use_dummy_text=(args.use_dummy_text if 'use_dummy_text' in args.keys() else False),
        )
        dataloader = make_dataloader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    elif args.data_type == 'interhuman':
        dataset = InterHumanDataset(
            max_len=args.max_len,
            mean_path=(args.mean_path if args.normalize else None),
            std_path=(args.std_path if args.normalize else None),
            trans_aug_rad=(args.trans_aug_rad if 'trans_aug_rad' in args.keys() else None),
            use_dummy_text=(args.use_dummy_text if 'use_dummy_text' in args.keys() else False),
        )
        dataloader = make_dataloader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    elif args.data_type == 'laion_pose':
        dataloader = LaionPose(
            val=val,
            use_dummy_text=args.pose_use_dummy_text,
            use_dummy_betas=args.pose_use_dummy_betas,
            normalize=args.normalize,
            mean_path=(args.mean_path if args.normalize else None),
            std_path=(args.std_path if args.normalize else None),
            randomly_rotate_pose=args.randomly_rotate_pose,
            trans_aug_rad=(args.trans_aug_rad if 'trans_aug_rad' in args.keys() else None),
            batch_size=batch_size,
            resampled=True,
            shuffle=args.shuffle
        )
    elif args.data_type == 'joint':
        dataloader = JointMotionPose(
            motion_max_len=args.motion_max_len,
            motion_use_dummy_text=args.motion_use_dummy_text,
            pose_use_dummy_betas=args.pose_use_dummy_betas,
            pose_use_dummy_text=args.pose_use_dummy_text,
            pose_val=val,
            data_split=args.data_split,
            normalize=args.normalize,
            mean_path=(args.mean_path if args.normalize else None),
            std_path=(args.std_path if args.normalize else None),
            amass_trans_aug_rad=args.amass_trans_aug_rad,
            interhuman_trans_aug_rad=args.interhuman_trans_aug_rad,
            webvid_trans_aug_rad=args.webvid_trans_aug_rad,
            pose_trans_aug_rad=args.pose_trans_aug_rad,
            batch_size=batch_size,
            shuffle=True,
        )
    elif args.data_type == 'motion_uncond':
        dataloader = JointMotionUncond(
            batch_size=batch_size,
            shuffle=True,
            # args for motion dataset
            motion_max_len=args.max_len,
            amass_trans_aug_rad=args.amass_trans_aug_rad,
            interhuman_trans_aug_rad=args.interhuman_trans_aug_rad,
            # shared mean and std
            normalize=args.normalize,
            mean_path=args.mean_path,
            std_path=args.std_path,
        )
    elif args.data_type == 'val':
        import importlib.resources as pkg_resources
        import pose_datasets as _pd_pkg
        _default_prompts = str(pkg_resources.files(_pd_pkg) / 'laion_viz_prompts.txt')
        dataset = ValPromptDataset(
            val_text=(args.val_text if 'val_text' in args.keys() else _default_prompts),
            mean_path=args.mean_path,
            std_path=args.std_path,       
        )
        dataloader = make_dataloader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    else:
        raise ValueError("Unrecognized dataset!!")

    return dataloader


def create_model(model_type, **kwargs):

    # multi person pose or single person motion
    if model_type == 'pose_net':
        return PoseNet(**kwargs)
    # joint training
    elif model_type == 'joint_net':
        return JointNet(**kwargs)
    elif model_type == 'joint_net_2_stage':
        return JointNet2Stage(**kwargs)
    elif model_type == 'pose2motion':
        return Pose2Motion(**kwargs)
    raise ValueError(f'Invalid model type: {model_type}')


def setup_eval_model(model_type, model_args, local_rank, checkpoint=None, guidance_param=None):
    model = create_model(model_type, **model_args)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
    model.to(local_rank)
    model.eval()
    if guidance_param is not None and guidance_param > 1.0:
        model = ClassifierFreeSampleModel(model, guidance_param=guidance_param)
    return model


def create_gaussian_diffusion(
    *,
    diffusion_steps=1000,
    noise_schedule="cosine",
    model_mean_type="xstart",
    timestep_respacing="",
    rescale_timesteps=False,
    use_ddim=False,
    interp_gamma=1.0,
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        rescale_timesteps=rescale_timesteps,
        use_ddim=use_ddim,
        interp_gamma=interp_gamma,
    )
