config = dict(
    # exp args
    out_dir="./outputs/motion",  # TODO: set to your desired output directory

    schedule_sampler="uniform",
    lr=1e-4,
    weight_decay=0.0,
    lr_anneal_steps=0,
    batch_size=32,
    microbatch=-1,  # -1 disables microbatches
    ema_rate="0.9999",  # comma-separated list of EMA values
    log_interval=100,
    save_interval=5000,
    resume_checkpoint="",

    dataset=dict(
        data_type='amass',
        max_len=61,
        normalize=False,
        mean_path=None,
        std_path=None,
        use_dummy_text=True,
        trans_aug_rad=(0.75, 0.75, 0.0),
    ),
    viz_gt=True,
    viz_frameskip=10,  # otherwise motion viz takes forever...
    guidance_param=None,

    # fid args
    eval_num_samples=1024,
    eval_batch_size=128,
    eval_interval=5000,  # None for no FID, otherwise frequency of fid calc
    get_eval_metrics=False,  # Set to True and provide eval_gt_*_file paths to enable FID evaluation
    eval_max_viz_samples=8,
    eval_gt_motion_file=None,
    eval_gt_motion_text_file=None,
    eval_pose_type="motion",

    # model args
    model_type='pose_net',
    model=dict(
        cond_mask_prob=0.0,
        dropout=0.1,
        latent_dim=512,
        num_heads=4,
        use_positional_encoding=True,
        use_text_condition=True,
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
