config = dict(
    # exp args
    out_dir=None,
    #out_dir=None,
    checkpoint=None,
    #checkpoint="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/multi-pose-v5/multi_pose_small_4_gpu/24-01-22-03-52-52/checkpoints/ema_0.9999_970000.pt",
    pose_type="joint",
    num_sample_frames=1,
    viz=True,
    frameskip=5,

    #batch_size=128,
    #eval_num_samples=1024,
    #viz_num_samples=16,

    batch_size=4,
    eval_num_samples=16,
    viz_num_samples=16,

    guidance_param=None,
    #guidance_param=1.75,

    #eval_gt_pose_file=None,
    #eval_gt_text_file=None,
    eval_gt_pose_file=None,
    eval_gt_text_file=None,
    eval_encoder_type="clpp",

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
    model_type='pose2motion',
    model=dict(
        cond_mask_prob=0.0,
        dropout=0.0,
        latent_dim=512,
        num_heads=4,
        use_pose_positional_encoding=True,
        use_pose_residual_format=True,
        pad_pos_emb_start=False,
        skip_pose_layers=False,
        freeze_pose_model=False,
        freeze_pose_transformer_only=False,
        use_text_condition=True,
        use_pose_condition=False,
        use_pose_only_first_frame=False,
        pose_model_ckpt=None,  # TODO: set to your trained checkpoint path
        #pose_model_ckpt='/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v1/multi-pose-v1/multi_pose_4_gpu/24-01-31-22-35-27/checkpoints/ema_0.9999_400000.pt',
    ),

    # diffusion args
    diffusion=dict(
        diffusion_steps=1000,
        noise_schedule="cosine",  # cosine or linear
        model_mean_type="xstart",  # "epsilon", "xstart". TODO: add "v"
        timestep_respacing="",
        rescale_timesteps=False,
        use_ddim=False,
        interp_gamma=1.0,
    ),
)
