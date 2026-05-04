import os
from datetime import datetime

import numpy as np
from scipy import linalg

import torch
import torch.distributed as dist

from data_collection.open_clip.src.training.clip_pose import setup_clpp, setup_clmp
from evaluations.sample_diffusion import sample_diffusion
from evaluations.sample_diffusion_2_stage import sample_diffusion_2_stage
from diffusion.dist_util import read_text_prompts, gather_sample_outputs
from pose_datasets.data_utils import process_translations_from_padded_poses, remove_pad_poses


def calculate_frechet_distance(encoding1, encoding2, eps=1e-6, max_dummy_fid=100.0):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- encoding 1/2: num_samples x dim
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.mean(encoding1, axis=0)
    sigma1 = np.cov(encoding1, rowvar=False)
    mu2 = np.mean(encoding2, axis=0)
    sigma2 = np.cov(encoding2, rowvar=False)
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    try:
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)
    except:
        # return dummy value if imaginary part is too big
        return max_dummy_fid


def evaluate_sim(pose_features, text_features, return_avg=True):
    pose_features_norm = pose_features / np.linalg.norm(pose_features, axis=-1, keepdims=True)
    text_features_norm = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)
    sim = (pose_features_norm * text_features_norm).sum(1)
    if return_avg:
        sim_avg = sim.mean()
        return float(sim_avg.item())
    else:
        return sim


def evaluate_diversity(pose_features, max_samples=300):
    num_samples = min(max_samples, pose_features.shape[0])
    rand_pose_feats_1 = pose_features[np.random.choice(pose_features.shape[0], num_samples, replace=False)]
    rand_pose_feats_2 = pose_features[np.random.choice(pose_features.shape[0], num_samples, replace=False)]
    return float(linalg.norm(rand_pose_feats_1 - rand_pose_feats_2, axis=1).mean())


def evaluate_r_precision(pose_features, text_features):

    assert pose_features.shape[0] >= 1024 and text_features.shape[0] >= 1024

    top_1_count = 0
    top_2_count = 0
    top_3_count = 0

    for group in range(32):
        pose_group = pose_features[group * 32 : (group + 1) * 32]  # [32, 768]
        text_group = text_features[group * 32 : (group + 1) * 32]  # [32, 768]
        for in_id in range(32):
            # compare each text with the other 31 texts in batch
            pose = pose_group[in_id:in_id+1]  # size [1, 768]
            sim = (pose @ text_group.transpose())[0]  # size [32]
            top_3_match = sim.argsort()[:-4:-1]
            if in_id == top_3_match[0]:
                top_1_count += 1
                top_2_count += 1
                top_3_count += 1
            elif in_id == top_3_match[1]:
                top_2_count += 1
                top_3_count += 1
            elif in_id == top_3_match[2]:
                top_3_count += 1

    return float(top_1_count / 1024), float(top_2_count / 1024), float(top_3_count / 1024)


def get_sim_diversity_rprec(pose_features, text_features):
    sim = evaluate_sim(pose_features, text_features)
    diversity = evaluate_diversity(pose_features)
    if pose_features.shape[0] >= 1024 and text_features.shape[0] >= 1024:
        r_top1, r_top2, r_top3 = evaluate_r_precision(pose_features, text_features)
    else:
        r_top1, r_top2, r_top3 = None, None, None
    return sim, diversity, r_top1, r_top2, r_top3


def get_encoder(encoder_type, device="cuda"):
    if encoder_type == 'clpp':  # for multi-person pose (eval for each frame)
        encoder = setup_clpp(device=device)
    elif encoder_type == 'clmp':  # for single-person motion (eval for each individual motion)
        encoder = setup_clmp(device=device)
    else:
        raise ValueError('Invalid "encoder_type".')
    return encoder


def format_report(
    fid=None,
    sim=None,
    diversity=None,
    r_top1=None,
    r_top2=None,
    r_top3=None,
    gt_sim=None,
    gt_diversity=None,
    gt_r_top1=None,
    gt_r_top2=None,
    gt_r_top3=None,
    encoder_type='CLPP'
):
    assert encoder_type in ['CLPP', 'CLMP']
    report = (
        "\n" + \
        f"{encoder_type} FID: {fid}\n" + \
        "\n" + \
        f"{encoder_type} Sim: {sim}\n" + \
        f"{encoder_type} Diversity: {diversity}\n" + \
        f"{encoder_type} R Prec Top 1: {r_top1}\n" + \
        f"{encoder_type} R Prec Top 2: {r_top2}\n" + \
        f"{encoder_type} R Prec Top 3: {r_top3}\n" + \
        "\n" + \
        f"{encoder_type} Sim Data: {gt_sim}\n" + \
        f"{encoder_type} Diversity Data: {gt_diversity}\n" + \
        f"{encoder_type} R Prec Top 1 Data: {gt_r_top1}\n" + \
        f"{encoder_type} R Prec Top 2 Data: {gt_r_top2}\n" + \
        f"{encoder_type} R Prec Top 3 Data: {gt_r_top3}" + \
        "\n"
    )
    return report


def get_eval_metrics(
    encoder,
    pose,
    text,
    gt_pose=None,
    gt_text=None,
    batch_size=None,
    device='cuda',
    get_gt_metrics=True,
):

    assert len(pose.shape) == 3, 'Eval format must be [batch_size, num_poses, 158] or [batch_size, num_frames, 158].'

    pose_features, text_features = \
        encoder.encode_pose_and_text(pose, text, batch_size=batch_size, device=device)

    if gt_pose is not None:
        if gt_text is None:
            gt_text = [''] * gt_pose.shape[0]
        gt_pose_features, gt_text_features = \
            encoder.encode_pose_and_text(gt_pose, gt_text, batch_size=batch_size, device=device)

    if gt_pose is not None and pose_features.shape[0] >= 768:
        fid = calculate_frechet_distance(pose_features.numpy(), gt_pose_features.numpy(), eps=1e-6)
    else:
        fid = None

    sim, diversity, r_top1, r_top2, r_top3 = \
        get_sim_diversity_rprec(pose_features.numpy(), text_features.numpy())
    if gt_pose is not None and get_gt_metrics:
        gt_sim, gt_diversity, gt_r_top1, gt_r_top2, gt_r_top3 = \
            get_sim_diversity_rprec(gt_pose_features.numpy(), gt_text_features.numpy())
    else:
        gt_sim = None
        gt_diversity = None
        gt_r_top1 = None
        gt_r_top2 = None
        gt_r_top3 = None

    return {
        "fid": fid,
        "sim": sim,
        "diversity": diversity,
        "r_top1": r_top1,
        "r_top2": r_top2,
        "r_top3": r_top3,
        "gt_sim": gt_sim,
        "gt_diversity": gt_diversity,
        "gt_r_top1": gt_r_top1,
        "gt_r_top2": gt_r_top2,
        "gt_r_top3": gt_r_top3,
    }


def get_eval_metrics_avg(
    encoder,
    encoder_type,
    pose,
    text,
    gt_pose=None,
    gt_text=None,
    batch_size=None,
    device='cuda',
    eval_frame_skip=15,  # for pose model, the 
):

    assert encoder_type in ['clpp', 'clmp']
    if encoder_type == 'clmp':
        assert len(pose.shape) == 3

    if len(pose.shape) == 3:
        if encoder_type == 'clpp':
            pose = pose.unsqueeze(1)
        if encoder_type == 'clmp':
            # create dummy pose dim.
            # all pose data should be shifted to batch dim before using this fn,
            # so iterating over the pose axis should only involve a single metric calculation.
            pose = pose.unsqueeze(2)

    if encoder_type == 'clpp':
        iter_axis = 1
    elif encoder_type == 'clmp':
        iter_axis = 2
    else:
        raise ValueError('Invalid "encoder_type".')

    metric_dict_all = []
    if encoder_type == 'clpp':
        # evaluate frames that are spaced separated by a fixed time interval
        eval_range = range(0, pose.shape[iter_axis], eval_frame_skip)
    else:
        # eval pose of each person
        eval_range = range(pose.shape[iter_axis])

    for idx in eval_range:
        if iter_axis == 1:
            pose_slice = pose[:, idx]
        else:
            pose_slice = pose[:, :, idx]
        metric_dict = get_eval_metrics(
                            encoder=encoder,
                            pose=pose_slice,
                            text=text,
                            gt_pose=gt_pose,
                            gt_text=gt_text,
                            batch_size=batch_size,
                            device=device,
                            get_gt_metrics=(idx == 0),
                        )
        metric_dict_all.append(metric_dict)

    metric_dict_avg = {'encoder_type': encoder_type.upper()}
    for key in metric_dict_all[0].keys():
        if key.startswith('gt_'):
            # gt values only come from first dict
            metric_dict_avg[key] = metric_dict_all[0][key]
        else:
            vals_are_not_None = True
            for j in range(len(metric_dict_all)):
                if metric_dict_all[j][key] is None:
                    vals_are_not_None = False
            if vals_are_not_None:
                # average sample metrics across frames/poses
                metric_dict_avg[key] = np.mean(np.array(
                    [metric_dict_all[j][key] for j in range(len(metric_dict_all))]
                ))
            else:
                metric_dict_avg[key] = None

    metric_report = format_report(**metric_dict_avg)

    return metric_report


def sample_from_trainer(trainer):
    if trainer.model_1st_stage is None:
        pose, cond = sample_diffusion(
            model=(
                trainer.cfg_model if hasattr(trainer, 'cfg_model') and trainer.cfg_model is not None
                else trainer.model
            ),
            diffusion=trainer.diffusion,
            num_samples=(trainer.eval_num_samples // dist.get_world_size()),
            local_rank=trainer.local_rank,
            dataloader=trainer.val_data,
            mean_path=trainer.mean_path,
            std_path=trainer.std_path,
            num_sample_frames=(
                trainer.eval_num_sample_frames if hasattr(trainer, 'eval_num_sample_frames')
                else None
            ),
        )
    else:
        pose, cond = sample_diffusion_2_stage(
            model_1st_stage=trainer.model_1st_stage,
            model_2nd_stage=(
                trainer.cfg_model if hasattr(trainer, 'cfg_model') and trainer.cfg_model is not None
                else trainer.model
            ),
            diffusion=trainer.diffusion,
            num_samples=(trainer.eval_num_samples // dist.get_world_size()),
            local_rank=trainer.local_rank,
            dataloader=trainer.val_data,
            mean_path=trainer.mean_path,
            std_path=trainer.std_path,
            num_sample_frames=(
                trainer.eval_num_sample_frames if hasattr(trainer, 'eval_num_sample_frames')
                else None
            ),
        )
    return pose, cond


def evaluate_clpp(
    encoder,
    poses,
    text_prompts,
    gt_poses,
    gt_text_prompts,
    num_frames,
    num_poses,
    batch_size,
    local_rank,
):
    # For CLPP eval, center each frame separately (to match expected input for eval model).
    poses_centered = process_translations_from_padded_poses(
                poses,
                num_frames=num_frames,
                num_poses=num_poses,
                pose_view=True,
                trans_aug_rad=None,
                single_motion_mode=False,
                center_first_frame=True,
                center_trans=True,
            )
    metric_report = get_eval_metrics_avg(
                        encoder=encoder,
                        encoder_type='clpp',
                        pose=poses_centered,
                        text=text_prompts,
                        gt_pose=gt_poses,
                        gt_text=gt_text_prompts,
                        batch_size=batch_size,
                        device=local_rank,
                    )
    return metric_report


def get_non_empty_motions(poses, text_prompts, num_poses):
    if len(poses.shape) == 4:
        poses_out = []
        text_prompts_out = []
        for idx in range(poses.shape[0]):
            pose_idx = poses[idx]
            # remove the empty poses
            pose_idx_nonempty = remove_pad_poses(pose_idx, num_frames=None, num_poses=num_poses[idx])
            # move pose to batch dim
            pose_idx_nonempty = pose_idx_nonempty.permute(1, 0, 2).contiguous()
            poses_out.append(pose_idx_nonempty)
            # repeat the text prompt according to the number of people
            text_prompts_out += [text_prompts[idx]] * num_poses[idx]
        # return 3D tensor [num_poses, num_frames, pose_feat] tensor of non-empty motions
        poses_out = torch.cat(poses_out, 0)
    else:
        poses_out = poses
        text_prompts_out = text_prompts
    return poses_out, text_prompts_out


def evaluate_clmp(
    encoder,
    poses,
    text_prompts,
    gt_poses,
    gt_text_prompts,
    num_frames,
    num_poses,
    batch_size,
    local_rank,
):
    # For CLMP eval, center the first frame (to match expected input for eval model).
    poses_centered = process_translations_from_padded_poses(
                poses,
                num_frames=num_frames,
                num_poses=num_poses,
                pose_view=False,
                trans_aug_rad=None,
                single_motion_mode=True,
                center_first_frame=True,
                center_trans=True,
            )
    poses_centered_nonempty, text_prompts_nonempty = \
        get_non_empty_motions(poses_centered, text_prompts, num_poses)
    metric_report = get_eval_metrics_avg(
                        encoder=encoder,
                        encoder_type='clmp',
                        pose=poses_centered_nonempty,
                        text=text_prompts_nonempty,
                        gt_pose=gt_poses,
                        gt_text=gt_text_prompts,
                        batch_size=batch_size,
                        device=local_rank,
                    )
    return metric_report


def evaluate_from_trainer(trainer):

    pose, cond = sample_from_trainer(trainer)

    pose_all, text_prompts_all, num_frames_all, num_poses_all = \
        gather_sample_outputs(pose, cond, trainer.local_rank)

    if trainer.get_eval_metrics and trainer.eval_gt_pose_file is not None and dist.get_rank() == 0:
        metric_report_clpp = evaluate_clpp(
                            encoder=trainer.encoder_clpp,
                            poses=pose_all,
                            text_prompts=text_prompts_all,
                            gt_poses=trainer.gt_pose,
                            gt_text_prompts=trainer.gt_pose_text,
                            num_frames=num_frames_all,
                            num_poses=num_poses_all,
                            batch_size=trainer.eval_batch_size,
                            local_rank=trainer.local_rank,
                        )
    if trainer.get_eval_metrics and trainer.eval_gt_motion_file is not None and dist.get_rank() == 0:
        metric_report_clmp = evaluate_clmp(
                            encoder=trainer.encoder_clmp,
                            poses=pose_all,
                            text_prompts=text_prompts_all,
                            gt_poses=trainer.gt_motion,
                            gt_text_prompts=trainer.gt_motion_text,
                            num_frames=num_frames_all,
                            num_poses=num_poses_all,
                            batch_size=trainer.eval_batch_size,
                            local_rank=trainer.local_rank,
                        )
    if trainer.get_eval_metrics and dist.get_rank() == 0:
        if trainer.eval_gt_motion_file is None:
            metric_report = metric_report_clpp
        elif trainer.eval_gt_pose_file is None:
            metric_report = metric_report_clmp
        else:
            # concatenate the metric reports for a joint report
            metric_report = metric_report_clpp + metric_report_clmp
    dist.barrier()

    if trainer.get_eval_metrics and dist.get_rank() == 0:
        return pose, cond, metric_report
    else:
        return pose, cond, None


def print_and_write(line, file_name):
    print(line)
    if file_name is not None:
        with open(file_name, 'a') as f:
            f.write(line)


def get_eval_metrics_from_saved_files(
    sample_pose_path,
    sample_text_path,
    gt_pose_path=None,
    gt_pose_text_path=None,
    gt_motion_path=None,
    gt_motion_text_path=None,
    batch_size=50,
):

    assert gt_pose_path is not None or gt_motion_path is not None
    assert gt_pose_text_path is not None or gt_motion_text_path is not None

    if gt_pose_path is not None:
        encoder_clpp = get_encoder('clpp')
    if gt_motion_path is not None:
        encoder_clmp = get_encoder('clmp')

    # read the saved sample data
    if sample_pose_path.endswith('.npy'):
        sample_pose = torch.tensor(np.load(sample_pose_path)).cuda()
        num_frames = sample_pose.shape[1] * torch.ones([sample_pose.shape[0]]).long().cuda()
        num_poses = sample_pose.shape[2] * torch.ones([sample_pose.shape[0]]).long().cuda()
    elif sample_pose_path.endswith('.npz'):
        sample_npz = np.load(sample_pose_path)
        sample_pose = torch.tensor(sample_npz['sample']).cuda()
        num_frames = (
            torch.tensor(sample_npz['num_frames']).cuda() if 'num_frames' in sample_npz.files
            else None
        )
        num_poses = (
            torch.tensor(sample_npz['num_poses']).cuda() if 'num_poses' in sample_npz.files
            else None
        )
    else:
        raise ValueError('"sample_pose_path" must end with either .npy or .npz')
    sample_text = read_text_prompts(sample_text_path)

    if gt_pose_path is not None:
        gt_pose = torch.tensor(np.load(gt_pose_path)).cuda()
        gt_pose_text = read_text_prompts(gt_pose_text_path)

    if gt_motion_path is not None:
        gt_motion = torch.tensor(np.load(gt_motion_path)).cuda()
        gt_motion_text = read_text_prompts(gt_motion_text_path)

    if gt_pose_path is not None:
        metric_report_clpp = evaluate_clpp(
                                encoder=encoder_clpp,
                                poses=sample_pose,
                                text_prompts=sample_text,
                                gt_poses=gt_pose,
                                gt_text_prompts=gt_pose_text,
                                num_frames=num_frames,
                                num_poses=num_poses,
                                batch_size=batch_size,
                                local_rank='cuda',
                            )
    if gt_motion_path is not None:
        metric_report_clmp = evaluate_clmp(
                                encoder=encoder_clmp,
                                poses=sample_pose,
                                text_prompts=sample_text,
                                gt_poses=gt_motion,
                                gt_text_prompts=gt_motion_text,
                                num_frames=num_frames,
                                num_poses=num_poses,
                                batch_size=batch_size,
                                local_rank='cuda',
                            )

    if gt_motion_path is None:
        metric_report = metric_report_clpp
    elif gt_pose_path is None:
        metric_report = metric_report_clmp
    else:
        # concatenate the metric reports for a joint report
        metric_report = metric_report_clpp + metric_report_clmp

    out_dir = os.path.dirname(sample_pose_path)
    stamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    metric_out_file = os.path.join(out_dir, f'eval-{stamp}.txt')

    print_and_write(metric_report, file_name=metric_out_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_pose_path", type=str, default="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-v1/pose2motion_2_stage_8_gpu/24-02-26-02-10-20/samples/pose2motion_2_stage_8_gpu/24-02-28-23-55-51/samples_1024x61x10x158.npz")
    parser.add_argument("--sample_text_path", type=str, default="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-v1/pose2motion_2_stage_8_gpu/24-02-26-02-10-20/samples/pose2motion_2_stage_8_gpu/24-02-28-23-55-51/samples_1024x61x10x158.txt")
    parser.add_argument("--gt_pose_path", type=str, default="/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/laion_pose_v4/data_samples_10000x10x158.npy")
    parser.add_argument("--gt_pose_text_path", type=str, default="/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/laion_pose_v4/data_samples_10000x10x158.txt")
    parser.add_argument("--gt_motion_path", type=str, default="/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/amass_interhuman_v1/data_samples_4096x61x158.npy")
    parser.add_argument("--gt_motion_text_path", type=str, default="/nfs/USRCSEA/IVA/Experiments/frechet-distance/ref_samples/amass_interhuman_v1/data_samples_4096x61x158.txt")
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    get_eval_metrics_from_saved_files(
        args.sample_pose_path,
        args.sample_text_path,
        gt_pose_path=(
            args.gt_pose_path if not args.gt_pose_path == "" else None
        ),
        gt_pose_text_path=(
            args.gt_pose_text_path if not args.gt_pose_text_path == "" else None
        ),
        gt_motion_path=(
            args.gt_motion_path if not args.gt_motion_path == "" else None
        ),
        gt_motion_text_path=(
            args.gt_motion_text_path if not args.gt_motion_text_path == "" else None
        ),
        batch_size=args.batch_size,
    )
