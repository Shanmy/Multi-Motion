import os
import argparse
from datetime import datetime

import torch

import webdataset as wds

import sys
sys.path.append('./open_clip/src/')
from training.clip_pose import CLPP, setup_clpp
sys.path.append('../pose_datasets/')
from laion_pose_loader import LaionPoseWDS
from clip_tokenizer import tokenize as clip_tokenizer
from data_utils import get_smpl_params_from_pose
sys.path.append('../viz/')
from viz_bev import viz_render_bev
sys.path.append('../diffusion/')
from evaluations.fid import evaluate_clip_sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_pose_ckpt", type=str, default="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/clip_pose/pose/2024_01_12-04_56_51-model_RN50-lr_0.0005-b_320-j_4-p_amp/checkpoints/epoch_100.pt")
    parser.add_argument("--pose_data_files", type=str, default="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/v2/{00000..00714}.tar")
    parser.add_argument("--laion_path", type=str, default="/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data4/")
    parser.add_argument("--out_path", type=str, default="out_clip_sorting_viz/")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--thresh", type=float, default=0.15, help="minimum clip pose alignment to retain sample")
    parser.add_argument("--burnin_batches", type=int, default=200)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--render", type=lambda x: (str(x).lower() == 'true'), default=True, help="render visualizations of best/worst samples")
    parser.add_argument("--save_tar_samples", type=lambda x: (str(x).lower() == 'true'), default=False, help="save best samples to tarfile")
    parser.add_argument("--tar_save_path", type=str, default="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/v_test/test_data_2.tar")
    args = parser.parse_args()

    # folder to save viz results
    out_path = os.path.join(args.out_path, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    os.makedirs(out_path)
    os.makedirs(os.path.join(out_path, 'best_samples'))
    os.makedirs(os.path.join(out_path, 'worst_samples'))

    pose_data = LaionPoseWDS(
                    urls=args.pose_data_files,
                    max_poses=10,
                    process_trans=True,
                    use_dummy_betas=True,
                    randomly_rotate_pose=False,
                )
    pose_dataloader = pose_data.wds_loader(
                            batch_size=args.batch_size,
                            resampled=True,
                            concat_vars=True,
                            yield_cond_dict=False,
                            yield_num_poses=True,
                            yield_trans_shift=True,
                            yield_key=True,
                        )
    data_iterator = iter(pose_dataloader)
    for _ in range(args.burnin_batches):
        _ = next(data_iterator)

    model = setup_clpp()

    for i, (pose, laion_text, blip2_text, instruct_blip_text, num_poses, trans_shift, key) in enumerate(data_iterator):

        # set up all pairings for all three text captions
        pose = pose.repeat(3, 1, 1)
        trans_shift = trans_shift.repeat(3, 1, 1)
        num_poses = num_poses.repeat(3)
        texts = laion_text + blip2_text + instruct_blip_text
        key = key * 3

        # prepare alternate format of pose for BEV viz
        betas, thetas, trans = get_smpl_params_from_pose(pose, reverse_align_pose=True, trans_shift=trans_shift)

        print(f'Scanning batch {i+1}')
        if i >= args.max_batches:
            break
        pose = pose.cuda()
        with torch.no_grad():
            pose_features, text_features = \
                model.encode_pose_and_text(pose, texts, batch_size=args.batch_size, device='cuda')
            sim = evaluate_clip_sim(pose_features, text_features, return_avg=False)
        sim_sorted = sim.sort().values.cpu().numpy()
        sorted_indices = sim.sort().indices.cpu().numpy()

        best_samples = list(sorted_indices[sim_sorted >= args.thresh])
        best_sim = list(sim_sorted[sim_sorted >= args.thresh])
        best_samples.reverse()
        best_sim.reverse()
        worst_samples = list(sorted_indices[sim_sorted < args.thresh])
        worst_sim = list(sim_sorted[sim_sorted < args.thresh])

        best_texts = [texts[j] for j in best_samples]
        worst_texts = [texts[j] for j in worst_samples]
        best_keys = [key[j] for j in best_samples]
        worst_keys = [key[j] for j in worst_samples]

        best_betas = betas[best_samples]
        best_thetas = thetas[best_samples]
        best_trans = trans[best_samples]
        worst_betas = betas[worst_samples]
        worst_thetas = thetas[worst_samples]
        worst_trans = trans[worst_samples]

        best_num_poses = num_poses[best_samples]
        worst_num_poses = num_poses[worst_samples]

        for j in range(best_thetas.shape[0]):
            if args.render:
                viz_render_bev(
                    betas=best_betas[j],
                    thetas=best_thetas[j],
                    trans=best_trans[j],
                    viz_text=f"text: {best_texts[j]} \n sim: {best_sim[j]}",
                    key=best_keys[j],
                    out_path=os.path.join(out_path, "best_samples"),
                    laion_path=args.laion_path,
                    num_poses=best_num_poses[j],
                    id_str=str(best_sim[j]),
                )
                viz_render_bev(
                    betas=worst_betas[j],
                    thetas=worst_thetas[j],
                    trans=worst_trans[j],
                    viz_text=f"text: {worst_texts[j]} \n sim: {worst_sim[j]}",
                    key=worst_keys[j],
                    out_path=os.path.join(out_path, "worst_samples"),
                    laion_path=args.laion_path,
                    num_poses=worst_num_poses[j],
                    id_str=str(worst_sim[j]),
                )
            if args.save_tar_samples:
                with wds.TarWriter(args.tar_save_path) as sink:
                    sample = {
                        "__key__": best_keys[j],
                        "beta.npy": best_betas[j].cpu().numpy(),
                        "theta.npy": best_thetas[j].cpu().numpy(),
                        "trans.npy": best_trans[j].cpu().numpy(),
                        "text.json": dict(blip_text=best_texts[j], laion_text=""),
                    }
                    sink.write(sample)
