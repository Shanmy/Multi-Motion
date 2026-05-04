config = dict(
    # exp args
    out_dir=None,
    checkpoint=None,  # TODO: set to your trained checkpoint path
    pose_type="motion",
    viz=True,
    frameskip=10,

    batch_size=128,
    eval_num_samples=1024,
    viz_num_samples=16,

    guidance_param=None,
    #guidance_param=1.75,

    get_eval_metrics=False,  # Set to True and provide eval_gt_*_file paths to enable FID evaluation
    #eval_gt_pose_file=None,
    #eval_gt_pose_text_file=None,
    eval_gt_motion_file=None,
    eval_gt_motion_text_file=None,
    #eval_gt_motion_file=None,
    #eval_gt_motion_text_file=None,

    dataset=dict(
        data_type='motion_uncond',
        max_len=61,
        normalize=False,
        mean_path=None,
        std_path=None,
        amass_trans_aug_rad=(1.0, 1.0, 0.0),
        interhuman_trans_aug_rad=(1.0, 1.0, 0.0),
    ),

    # model args
    model_type='pose_net',
    model=dict(
        cond_mask_prob=0.0,
        latent_dim=512,
        num_heads=4,
        use_positional_encoding=True,
        use_text_condition=True,
        use_pose_condition=True,
        inpaint_gt_first_pose=True,
        use_layer_residual_format=True,
    ),

    # diffusion args
    diffusion=dict(
        diffusion_steps=1000,
        noise_schedule="cosine",  # cosine or linear
        model_mean_type="xstart",  # "epsilon", "xstart". TODO: add "v"
        timestep_respacing="",
        rescale_timesteps=False,
        use_ddim=False,
        interp_gamma=0.5,
    ),
)
