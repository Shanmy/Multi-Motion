import os

from diffusion.dist_util import setup_exp
from base_sample import main as sample


CFG_COEFS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
CFG_COEFS_STAGE_1 = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
#CFG_COEFS_STAGE_1 = [None]
VIZ = True
EVAL = True

config, world_rank, local_rank, world_size = setup_exp(__file__)
exp_path_base = config['out_dir']
config['viz'] = VIZ
config['get_eval_metrics'] = EVAL

for cfg_coef in CFG_COEFS:
    for cfg_coef_stage_1 in CFG_COEFS_STAGE_1:
        config['guidance_param'] = cfg_coef
        if cfg_coef_stage_1 is not None:
            config['guidance_param_1st_stage'] = cfg_coef_stage_1
            if 'cond_model_args' in config.keys():
                config['cond_model_args']['pose_guidance_param'] = cfg_coef_stage_1
        config['out_dir'] = os.path.join(exp_path_base, f'cfg_{cfg_coef}_stage1_{cfg_coef_stage_1}')
        if world_rank == 0:
            for subfolder in ('checkpoints', 'viz', 'log', 'code'):
                os.makedirs(os.path.join(config['out_dir'], subfolder))
        sample(config, world_rank, local_rank, world_size)
