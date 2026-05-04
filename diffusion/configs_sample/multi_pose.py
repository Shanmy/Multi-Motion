config = dict(
    # exp args
    out_dir=None,
    checkpoint=None,  # TODO: set to your trained checkpoint path
    #checkpoint='/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/multi-pose-v1/multi_pose_4_gpu/24-02-25-22-22-36/checkpoints/ema_0.9999_490000.pt',
    pose_type="multipose",
    viz=True,
    frameskip=10,

    batch_size=128,
    eval_num_samples=1024,
    viz_num_samples=16,

    #guidance_param=None,
    guidance_param=1.75,

    get_eval_metrics=False,  # Set to True and provide eval_gt_*_file paths to enable FID evaluation
    eval_gt_pose_file=None,
    eval_gt_pose_text_file=None,

    dataset=dict(
        data_type='laion_pose',
        pose_use_dummy_text=False,
        pose_use_dummy_betas=True,
        randomly_rotate_pose=True,
        normalize=False,
        mean_path=None,
        std_path=None,
    ),

    # model args
    model_type='pose_net',
    model=dict(
        cond_mask_prob=0.0,
        latent_dim=512,
        num_heads=4,
        use_positional_encoding=False,
        use_text_condition=True,
        use_layer_residual_format=True,
    ),

    # diffusion args
    diffusion=dict(
        diffusion_steps=1000,
        noise_schedule="cosine",  # cosine or linear
        model_mean_type="xstart",  # "epsilon", "xstart". TODO: add "v"
        timestep_respacing="256",
        rescale_timesteps=False,
        use_ddim=True,
        interp_gamma=0.5,
    ),
)
