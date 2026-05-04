import torch

from diffusion.script_util import setup_eval_model
from models.net_utils import full_to_motion, motion_to_full, full_to_pose, pose_to_full


def repeat_text(text, num_reps):
    return [ele for ele in text for i in range(num_reps)]

def repeat_1D_tensor(tensor, num_reps):
    return tensor.view(-1, 1).repeat(1, num_reps).view(-1).contiguous()

def repeat_2D_tensor(tensor, num_reps):
    tensor_expand = tensor.view(tensor.shape[0], 1, tensor.shape[1])
    tensor_repeat = tensor_expand.repeat(1, num_reps, 1)
    tensor_reshape = tensor_repeat.view(-1, tensor.shape[1]).contiguous()
    return tensor_reshape

# this function will apply guidance to a joint model
# a pose-only model can provide guidance to each frame
# a single-person motion model can provide guidance to each individual motion

def get_denoised_fn(cond_model_dict, model_denoised_fn, pose_guidance_frameskip=30):

    def prepare_pose_conds(tstep, cond, max_num_frames):
        # repeat timesteps and conditions over frames
        tstep_pose = repeat_1D_tensor(tstep, max_num_frames)
        cond_pose = {
            "text_encoding": repeat_2D_tensor(cond["text_encoding"], max_num_frames),
            "num_poses": repeat_1D_tensor(cond["num_poses"], max_num_frames),
        }
        return tstep_pose, cond_pose

    def prepare_motion_conds(tstep, cond, max_num_poses, uncond_motion=True):
        assert uncond_motion, "Need to build method for getting single-person caption from multi-person caption."
        # repeat timesteps and conditions over frames
        tstep_motion = repeat_1D_tensor(tstep, max_num_poses)
        if uncond_motion:
            text_encoding = cond["text_encoding_uncond"]
        else:
            raise ValueError("Need to implement single-person captions derived from group caption.")
        cond_motion = {
            "text_encoding": repeat_2D_tensor(text_encoding, max_num_poses),
            "lengths": repeat_1D_tensor(cond["lengths"], max_num_poses),
        }
        if (
            ('motion_uncond_model' in cond_model_dict.keys() and cond_model_dict['motion_uncond_model'].use_pose_condition and uncond_motion) or
            ('motion_cond_model' in cond_model_dict.keys() and cond_model_dict['motion_cond_model'].use_pose_condition and not uncond_motion)
        ):
            pose_cond = cond['pose_cond'].view(-1, cond['pose_cond'].shape[-1]).contiguous()
            cond_motion['pose_cond'] = pose_cond[:, :(pose_cond.shape[-1] // 2)]
        return tstep_motion, cond_motion

    def cond_fn(x, t, xstart_pred, **cond):
        assert len(x.shape) == 4, 'Guidance is only set up for multi-person motion models.'
        out = torch.zeros_like(x)
        if 'pose_model' in cond_model_dict.keys() and cond_model_dict['pose_model_coef'] > 0:
            # condition score from pose-only model
            x_pose = full_to_pose(x)
            t_pose, cond_pose = prepare_pose_conds(t, cond, x.shape[1])
            out_pose = cond_model_dict['pose_model'](x_pose, t_pose, **cond_pose)
            out_pose = pose_to_full(out_pose, batch_size=x.shape[0])
            # only use guidance term for sparse frames
            for frame_idx in range(out_pose.shape[1]):
                if not frame_idx % pose_guidance_frameskip == 0:
                    out_pose[:, frame_idx] = xstart_pred[:, frame_idx]
            # 'pose_model_coef' controls guidance strength
            out += cond_model_dict['pose_model_coef'] * out_pose
        if (
            'motion_uncond_model' in cond_model_dict.keys() and
            cond_model_dict['motion_uncond_model_coef'] > 0
        ):
            # condition score from unconditional single-person motion model
            x_motion_uncond = full_to_motion(x)
            t_motion_uncond, cond_motion_uncond = \
                prepare_motion_conds(t, cond, x.shape[2], uncond_motion=True)
            out_motion_uncond = cond_model_dict['motion_uncond_model'](
                                        x_motion_uncond, t_motion_uncond, **cond_motion_uncond
                                    )
            out_motion_uncond = motion_to_full(out_motion_uncond, batch_size=x.shape[0])
            # 'motion_uncond_model_coef' controls guidance strength
            out += cond_model_dict['motion_uncond_model_coef'] * out_motion_uncond
        if (
            'motion_cond_model' in cond_model_dict.keys() and
            cond_model_dict['motion_cond_model_coef'] > 0
        ):
            # condition score from  text conditional single-person motion model
            x_motion_cond = full_to_motion(x)
            t_motion_cond, cond_motion_cond = \
                prepare_motion_conds(t, cond, x.shape[2], uncond_motion=False)
            out_motion_cond = cond_model_dict['motion_cond_model'](
                                        x_motion_cond, t_motion_cond, **cond_motion_cond
                                    )
            out_motion_cond = motion_to_full(out_motion_cond, batch_size=x.shape[0])
            # 'motion_cond_model_coef' controls guidance strength
            out += cond_model_dict['motion_cond_model_coef'] * out_motion_cond
        return out

    def denoised_fn(xstart_pred, x_t, tstep, cond_dict):
        if cond_model_dict is not None:
            xstart_pred_cond = cond_fn(x_t, tstep, xstart_pred, **cond_dict)
            if cond_model_dict['average_across_models']:
                # average diffusion predictions with weights that sum to 1
                total_cond_weight = 0
                if 'pose_model_coef' in cond_model_dict.keys():
                    total_cond_weight += cond_model_dict['pose_model_coef']
                if 'motion_uncond_model_coef' in cond_model_dict.keys():
                    total_cond_weight += cond_model_dict['motion_uncond_model_coef']
                if 'motion_cond_model_coef' in cond_model_dict.keys():
                    total_cond_weight += cond_model_dict['motion_cond_model_coef']
                assert total_cond_weight <= 1.0
                orig_weight = 1.0 - total_cond_weight
                xstart_pred = orig_weight * xstart_pred + xstart_pred_cond
            else:
                xstart_pred = xstart_pred + xstart_pred_cond
        if model_denoised_fn is not None:
            xstart_pred = model_denoised_fn(batch=xstart_pred, cond_dict=cond_dict)
        return xstart_pred

    return denoised_fn


# this function will prepare helper models used in denoised_fn

def get_cond_model_dict(args, local_rank):
    cond_model_dict = {'average_across_models': args.average_across_models}
    if 'pose_model' in args.keys() and args.pose_model is not None:
        cond_model_dict["pose_model"] = setup_eval_model(
            model_type=args.pose_model_type,
            model_args=args.pose_model,
            local_rank=local_rank,
            checkpoint=(
                args.pose_path
                if 'pose_path' in args.keys()
                else None
            ),
            guidance_param=(
                args.pose_guidance_param
                if 'pose_guidance_param' in args.keys()
                else None
            )
        )
        cond_model_dict["pose_model_coef"] = args.pose_model_coef
    if (
        'motion_uncond_model' in args.keys() 
        and args.motion_uncond_model is not None
    ):
        cond_model_dict['motion_uncond_model'] = setup_eval_model(
            model_type=args.motion_uncond_model_type,
            model_args=args.motion_uncond_model,
            local_rank=local_rank,
            checkpoint=(
                args.motion_uncond_path
                if 'motion_uncond_path' in args.keys()
                else None
            ),
            guidance_param=None,  # no guidance for uncond model
        )
        cond_model_dict["motion_uncond_model_coef"] = args.motion_uncond_model_coef
    if (
        'motion_cond_model' in args.keys() 
        and args.motion_cond_model is not None
    ):
        cond_model_dict['motion_cond_model'] = setup_eval_model(
            model_type=args.motion_cond_model_type,
            model_args=args.motion_cond_model,
            local_rank=local_rank,
            checkpoint=(
                args.motion_cond_path
                if 'motion_cond_path' in args.keys()
                else None
            ),
            guidance_param=(
                args.motion_cond_guidance_param
                if 'motion_cond_guidance_param' in args.keys()
                else None
            )
        )
        cond_model_dict["motion_cond_model_coef"] = args.motion_cond_model_coef
    return cond_model_dict
