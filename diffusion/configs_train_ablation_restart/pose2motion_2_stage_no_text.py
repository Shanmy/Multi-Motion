config = dict(
    # exp args
    out_dir="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-restart-v1",

    schedule_sampler="uniform",
    lr=1e-4,
    weight_decay=0.0,
    lr_anneal_steps=0,
    batch_size=20,
    microbatch=-1,  # -1 disables microbatches
    ema_rate="0.9999",  # comma-separated list of EMA values
    log_interval=100,
    save_interval=5000,
    resume_checkpoint="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-v1/pose2motion_2_stage_no_text_8_gpu/24-02-29-02-54-38/checkpoints/model200000.pt",

    dataset=dict(
        data_type='joint',
        data_split=(0.5, 0.0, 0.1, 0.0, 0.4),
        pose_use_dummy_betas=True,
        pose_use_dummy_text=False,
        motion_use_dummy_text=True,
        motion_max_len=61,
        normalize=False,
        mean_path=None,
        std_path=None,
        amass_trans_aug_rad=None,
        interhuman_trans_aug_rad=None,
        webvid_trans_aug_rad=None,
        pose_trans_aug_rad=(0.75, 0.75, 0.0),
    ),
    eval_dataset=dict(
        data_type='laion_pose',
        pose_use_dummy_text=False,
        pose_use_dummy_betas=True,
        randomly_rotate_pose=True,
        normalize=False,
        mean_path=None,
        std_path=None,
        trans_aug_rad=None,
    ),
    viz_gt=True,
    viz_frameskip=10,  # otherwise motion viz takes forever...
    #guidance_param=1.75,
    guidance_param=None,

    # fid args
    eval_num_samples=1024,
    eval_batch_size=128,
    eval_interval=5000,  # None for no FID, otherwise frequency of fid calc
    get_eval_metrics=True,
    eval_max_viz_samples=8,
    eval_gt_pose_file="/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/laion_pose_v4/data_samples_10000x10x158.npy",
    eval_gt_pose_text_file="/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/laion_pose_v4/data_samples_10000x10x158.txt",
    eval_gt_motion_file="/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/amass_interhuman_v1/data_samples_4096x61x158.npy",
    eval_gt_motion_text_file="/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/amass_interhuman_v1/data_samples_4096x61x158.txt",
    eval_num_sample_frames=61,
    eval_pose_type="joint",

    # model args
    model_type='pose2motion',
    model=dict(
        cond_mask_prob=0.1,
        dropout=0.1,
        latent_dim=512,
        num_heads=4,
        use_pose_positional_encoding=False,
        use_pose_residual_format=True,
        pad_pos_emb_start=False,
        skip_pose_layers=False,
        freeze_pose_model=True,
        freeze_pose_transformer_only=False,
        use_text_condition=True,
        use_pose_condition=True,
        use_pose_only_first_frame=False,
        pose_model_ckpt='/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/multi-pose-v1/multi_pose_4_gpu/24-02-25-22-22-36/checkpoints/ema_0.9999_490000.pt',
    ),

    # 1st stage model args for sampling eval
    model_type_1st_stage='pose_net',
    checkpoint_1st_stage='/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/multi-pose-v1/multi_pose_4_gpu/24-02-25-22-22-36/checkpoints/ema_0.9999_490000.pt',
    model_1st_stage=dict(
        cond_mask_prob=0.0,
        dropout=0.0,
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
        timestep_respacing="",
        rescale_timesteps=False,
        use_ddim=False,
        interp_gamma=0.5,
    ),
)
