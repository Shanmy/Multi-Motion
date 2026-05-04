config = dict(
    # exp args
    out_dir=None,
    checkpoint=None,  # TODO: set to your trained checkpoint path
    checkpoint_1st_stage=None,  # TODO: set to your trained checkpoint path
    pose_type="joint",
    viz=True,
    frameskip=2,

    batch_size=8,
    eval_num_samples=8,
    viz_num_samples=8,

    guidance_param_1st_stage=None,
    #guidance_param_1st_stage=1.75,
    guidance_param=None,
    #guidance_param=1.5,

    get_eval_metrics=False,  # Set to True and provide eval_gt_*_file paths to enable FID evaluation
    eval_gt_pose_file=None,
    eval_gt_pose_text_file=None,
    eval_gt_motion_file=None,
    eval_gt_motion_text_file=None,

    # for hand-selected prompts
    #dataset=dict(
    #    data_type='val',
    #    mean_path=None,
    #    std_path=None,
    #),

    # for eval metrics
    dataset=dict(
        data_type='laion_pose',
        pose_use_dummy_text=False,
        pose_use_dummy_betas=True,
        randomly_rotate_pose=True,
        normalize=False,
        mean_path=None,
        std_path=None,
        trans_aug_rad=None,
    ),
    num_sample_frames=61,

    # model args for 1st stage
    model_type_1st_stage='pose_net',
    model_1st_stage=dict(
        cond_mask_prob=0.0,
        dropout=0.0,
        latent_dim=512,
        num_heads=4,
        use_positional_encoding=False,
        use_text_condition=True,
        use_layer_residual_format=True,
    ),

    # model args for 2nd stage
    model_type='pose2motion',
    model=dict(
        cond_mask_prob=0.0,
        dropout=0.0,
        latent_dim=512,
        num_heads=4,
        use_pose_positional_encoding=False,
        use_pose_residual_format=True,
        pad_pos_emb_start=False,
        skip_pose_layers=False,
        freeze_pose_model=False,
        freeze_pose_transformer_only=False,
        use_text_condition=True,
        use_pose_condition=True,
        use_pose_only_first_frame=False,
        pose_model_ckpt=None,
        inpaint_gt_first_pose=True,
        #pose_model_ckpt='/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v1/multi-pose-v1/multi_pose_4_gpu/24-01-31-22-35-27/checkpoints/ema_0.9999_400000.pt',
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

    cond_model_args=dict(
        # True to average, false to sum
        average_across_models=True,

        pose_model_type="pose_net",
        pose_model_coef=0.1,
        pose_guidance_param=None,
        #pose_guidance_param=1.75,
        pose_path=None,  # TODO: set to your trained checkpoint path
        pose_model=dict(
            cond_mask_prob=0.0,
            latent_dim=512,
            num_heads=4,
            use_positional_encoding=False,
            use_text_condition=True,
            use_layer_residual_format=True,
        ),

        # model args
        motion_uncond_model_type='pose_net',
        motion_uncond_model_coef=0.0,
        motion_uncond_path=None,  # TODO: set to your trained checkpoint path
        motion_uncond_model=dict(
            cond_mask_prob=0.0,
            dropout=0.0,
            latent_dim=512,
            num_heads=4,
            use_positional_encoding=True,
            use_text_condition=True,
            use_pose_condition=True,
            inpaint_gt_first_pose=True,
            use_layer_residual_format=True,
        ),
    ),
)
