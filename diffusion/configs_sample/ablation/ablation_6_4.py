config = dict(
    # exp args
    out_dir="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mengyi-v3/ablation",
    checkpoint="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-v1/pose2motion_2_stage_webvid_freeze_pose_new_centered_8_gpu/24-03-06-06-00-54/checkpoints/ema_0.9999_120000.pt",
    checkpoint_1st_stage='/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/multi-pose-v1/multi_pose_all_8_gpu/24-03-05-11-17-36/checkpoints/ema_0.9999_230000.pt',
    #checkpoint_1st_stage="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/multi-pose-v1/multi_pose_4_gpu/24-02-25-22-22-36/checkpoints/ema_0.9999_490000.pt",
    pose_type="joint",
    viz=False,
    frameskip=2,

    batch_size=128,
    eval_num_samples=1024,
    viz_num_samples=20,

    #guidance_param_1st_stage=None,
    guidance_param_1st_stage=1.75,
    #guidance_param=None,
    guidance_param=1.5,

    get_eval_metrics=True,
    eval_gt_pose_file="/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/laion_pose_v4/data_samples_10000x10x158.npy",
    eval_gt_pose_text_file="/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/laion_pose_v4/data_samples_10000x10x158.txt",
    eval_gt_motion_file="/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/amass_interhuman_v1/data_samples_4096x61x158.npy",
    eval_gt_motion_text_file="/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/amass_interhuman_v1/data_samples_4096x61x158.txt",

    # for hand-selected prompts
    #dataset=dict(
    #    data_type='val',
    #    #val_text='/nfs/USRCSEA/IVA/Datasets/motion_baselines/intergen-1024x61x10x158.txt',
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
        shuffle=0
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
        use_pose_condition_first_frame=False,
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
        pose_model_coef=0.6,
        #pose_guidance_param=None,
        pose_guidance_param=1.75,
        pose_path="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/multi-pose-v1/multi_pose_4_gpu/24-02-25-22-22-36/checkpoints/ema_0.9999_490000.pt",
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
        motion_uncond_model_coef=0.4,

        # 1 stage motion guidance
        #motion_uncond_path='/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/motion-v1/motion_joint_uncond_4_gpu/24-02-29-02-47-53/checkpoints/ema_0.9999_200000.pt',
        #motion_uncond_model=dict(
        #    cond_mask_prob=0.0,
        #    dropout=0.0,
        #    latent_dim=512,
        #    num_heads=4,
        #    use_positional_encoding=True,
        #    use_text_condition=True,
        #    use_layer_residual_format=True,
        #),

        # 2 stage center frame guidance
        motion_uncond_path='/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/motion-v1/motion_joint_uncond_2_stage_centered_8_gpu/24-03-06-07-20-20/checkpoints/ema_0.9999_190000.pt',
        motion_uncond_model=dict(
            cond_mask_prob=0.0,
            dropout=0.0,
            latent_dim=512,
            num_heads=4,
            use_positional_encoding=True,
            use_text_condition=True,
            use_pose_condition=True,
            use_pose_condition_first_frame=False,
            inpaint_gt_first_pose=True,
            use_layer_residual_format=True,
        ),
    ),
)
