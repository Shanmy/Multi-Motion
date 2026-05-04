import os
from datetime import datetime

import torch

import sys
sys.path.append('./')
from models.pose_model import PoseNet
from diffusion.gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule
sys.path.append('../viz')
from viz_utils import viz_from_loader
sys.path.append('../pose_dataset')
from laion_pose_loader import LaionPose


save_dir = 'out_laion_viz'
save_dir = os.path.join(save_dir, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
batch_size = 10
ckpt = '/nfs/USRCSEA/IVA/Experiments/motion-diffusion/multi-pose-v2/multi_pose_4_gpu/23-12-18-06-10-29/checkpoints/model040000.pt'
mean_path = "../pose_datasets/stats_v1/stats_laion_pose/smpl_158_mean.npy"
std_path = "../pose_datasets/stats_v1/stats_laion_pose/smpl_158_std.npy"


pose_dataloader = LaionPose(
    normalize=True,
    batch_size=batch_size,
    resampled=True,
    concat_vars=True,
    yield_cond_dict=True,
    mean_path=mean_path,
    std_path=std_path,
)

model = PoseNet(cond_mask_prob=0.0)
model.load_state_dict(torch.load(ckpt))
model.to('cuda')

betas = get_named_beta_schedule(schedule_name="cosine", num_diffusion_timesteps=1000)
diffusion = GaussianDiffusion(betas=betas, model_mean_type='xstart', rescale_timesteps=False, use_ddim=False, interp_gamma=1.0)

poses, cond_dict = next(iter(pose_dataloader))
poses = poses.to('cuda')

viz_from_loader(
    poses=poses,
    cond_dict=cond_dict,
    save_dir=save_dir,
    mean_path=mean_path,
    std_path=std_path,
    unsqueeze_pose_dim=False,
    unsqueeze_motion_dim=True,
)

t = torch.randint(size=[poses.shape[0]], high=1000).to('cuda')
poses_sample_noisy = diffusion.q_sample(poses, t)

viz_from_loader(
    poses=poses_sample_noisy,
    cond_dict=cond_dict,
    save_dir=save_dir,
    mean_path=mean_path,
    std_path=std_path,
    unsqueeze_pose_dim=False,
    unsqueeze_motion_dim=True,
)

poses_denoised = model(poses_sample_noisy, t, **cond_dict)

viz_from_loader(
    poses=poses_denoised,
    cond_dict=cond_dict,
    save_dir=save_dir,
    mean_path=mean_path,
    std_path=std_path,
    unsqueeze_pose_dim=False,
    unsqueeze_motion_dim=True,
)
