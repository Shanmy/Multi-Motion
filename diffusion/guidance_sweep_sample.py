import os

from diffusion.dist_util import setup_exp
from base_sample import main as sample


#POSE_COEFS = [0.0, 0.033, 0.067, 0.1, 0.133, 0.167, 0.2, 0.23, 0.267, 0.3]
#MOTION_COEFS = [0.0, 0.033, 0.067, 0.1, 0.133, 0.167, 0.2, 0.23, 0.267, 0.3]
POSE_COEFS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MOTION_COEFS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
CUTOFF = 1.0
VIZ = True
EVAL = False

config, world_rank, local_rank, world_size = setup_exp(__file__)
exp_path_base = config['out_dir']
config['viz'] = VIZ
config['get_eval_metrics'] = EVAL

for motion_coef in MOTION_COEFS:
    for pose_coef in POSE_COEFS:
        if motion_coef + pose_coef <= CUTOFF:
            config['cond_model_args']['pose_model_coef'] = pose_coef
            config['cond_model_args']['motion_uncond_model_coef'] = motion_coef
            config['out_dir'] = os.path.join(exp_path_base, f'pose_{pose_coef}_motion_{motion_coef}')
            if world_rank == 0:
                for subfolder in ('checkpoints', 'viz', 'log', 'code'):
                    os.makedirs(os.path.join(config['out_dir'], subfolder))
            sample(config, world_rank, local_rank, world_size)
