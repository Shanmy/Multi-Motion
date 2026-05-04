import os
import shutil
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from textwrap import wrap

import trimesh
from trimesh import Trimesh

from viz.mp4_to_gif import convertFile
import torch

from smpl.smpl import SMPLA_parser
from pose_datasets.data_utils import unnormalize, remove_pad_poses, convert_6d_to_smpl, get_smpl_params_from_pose, get_pose_from_smpl_params

from glob import glob

from scipy.ndimage import gaussian_filter1d

_DEFAULT_SMPL_FACES = str(pathlib.Path(__file__).parent.parent / 'smpl' / 'smpl_faces.npy')

def get_trimesh(vertices, smpl_faces_file=_DEFAULT_SMPL_FACES):
    faces = np.load(smpl_faces_file)
    return Trimesh(vertices=vertices.tolist(), faces=faces)


def save_pose_obj(vertices, save_path):
    mesh = get_trimesh(vertices)
    with open(save_path, 'w') as fw:
        mesh.export(fw, 'obj')


def save_poses_from_verts(
    vertices, save_path_stem='', blender_viz=True, video_mode=True, text=None, viz_text=True, framerate=20
):

    # make new folder to save vertices (up to 100000 motions in one folder to avoid while loops)
    folder_id = 0
    for folder_id in range(100000):
        try:
            save_path_stem_folder = os.path.join(save_path_stem, 'sample_' + str(folder_id).zfill(5))
            os.makedirs(save_path_stem_folder)
            break
        except:
            pass

    # write text prompt to file, if there is one
    if text is not None:
        text_prompts_file = os.path.join(save_path_stem_folder, 'prompt.txt')
        with open(text_prompts_file, 'w') as f:
            f.write(text)

    # assume that vertices has shape: FxNxPx3
    # F is number of frames, N is number of people, P is number of vertices, 3 spatial coords
    assert len(vertices.shape) == 4, 'Vertices must be 4D tensor. Expand motion dim 0 or pose dim 1?'
    for frame_idx in range(vertices.shape[0]):
        # obj files for one multi-pose frame
        multipose_dir = os.path.join(save_path_stem_folder, f'frame_{str(frame_idx).zfill(5)}')
        os.makedirs(multipose_dir)
        obj_verts_list = []
        for pose_idx in range(vertices.shape[1]):
            save_path = os.path.join(multipose_dir, f'pose_{str(pose_idx).zfill(5)}.obj')
            save_pose_obj(vertices[frame_idx, pose_idx], save_path)
            # load obj npy and save all vertices
            loaded_mesh = trimesh.load(save_path, process=False)
            obj_verts = loaded_mesh.vertices
            if not (len(obj_verts_list) > 0 and obj_verts.shape[0] != obj_verts_list[0].shape[0]):
                # avoid case where loaded mesh has reduced vertices. should always use this in normal cases.
                obj_verts_list.append(obj_verts)
        obj_verts_npy = np.stack(obj_verts_list)
        obj_npy_path = f'{multipose_dir}/vertices.npy'
        np.save(obj_npy_path, obj_verts_npy)

    if blender_viz:
        # make png showing poses with blender
        create_visualization_motion_blender(save_path_stem_folder)
        if text is not None and viz_text:
            for obj_npy_path in glob(f"{save_path_stem_folder}/*/vertices.npy"):
                add_text_to_png(obj_npy_path, text)
        # edit png to add text if there is text
        
    if not (vertices.shape[0] > 1 and blender_viz and video_mode):
        # copy to outer folder
        png_file_path = os.path.join(os.path.dirname(obj_npy_path), 'vertices.png')
        png_copy_folder = os.path.dirname(os.path.dirname(obj_npy_path))
        png_copy_file = os.path.basename(png_copy_folder) + '.png'
        png_copy_path = os.path.join(os.path.dirname(png_copy_folder), png_copy_file)
        shutil.copyfile(png_file_path, png_copy_path)

    if vertices.shape[0] > 1 and blender_viz and video_mode:
        # animate pngs to make video
        create_video_from_pngs_ffmpeg(save_path_stem_folder, framerate)
        # convert mp4 to gif for easier viz, move to outer folder
        mp4_in_path = os.path.join(save_path_stem_folder, 'motion.mp4')
        save_id = os.path.basename(save_path_stem_folder)
        gif_out_path = os.path.join(os.path.dirname(save_path_stem_folder), save_id + '.gif')
        convertFile(inputpath=mp4_in_path, outputpath=gif_out_path)


def save_poses_from_smpl(
    betas,
    thetas,
    trans,
    save_path_stem='',
    smpla_parser=None,
    blender_viz=True,
    video_mode=True,
    text=None,
    framerate=20
):
    # each input is FxNxC, where F is number of frames, N is number of people, C is vector length
    assert len(betas.shape) == len(thetas.shape) == len(trans.shape) == 3
    if smpla_parser is None:
        smpla_parser = SMPLA_parser().to(betas.device)
    # smpl forward pass plus translation, applied to each pose separately then reshaped
    vertices, _ = smpla_parser.forward_trans(
                                betas.contiguous().view(-1, betas.shape[2]),
                                thetas.contiguous().view(-1, thetas.shape[2]),
                                trans.contiguous().view(-1, trans.shape[2]),
                                center_trans=False,
                            )
    # reshape to [F, N, P, 3], where P is the number of vertices of the smpl mesh
    vertices = vertices.view(thetas.shape[0], thetas.shape[1], vertices.shape[1], vertices.shape[2])
    # save
    save_poses_from_verts(
        vertices.detach().cpu(),
        save_path_stem=save_path_stem,
        blender_viz=blender_viz,
        video_mode=video_mode,
        text=text,
        framerate=framerate
    )


# visualize one multi-pose, single-motion, or multi-motion sample
# NOTE: function assumes dummy poses have already been removed
def viz_pose_sample(
    sample,  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
    save_dir,
    frameskip=None,
    text=None,
    mean_path=None,
    std_path=None,
    unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
    unsqueeze_motion_dim=False,  # set to True to pose sample with shape [num_poses, pose_dim]
    framerate=30
):

    assert sample.shape[-1] == 158, \
        'Pose dimension either must be 158 (11D betas, 144D thetas in 6d format, 3D trans).'

    if len(sample.shape) == 2:
        assert (unsqueeze_pose_dim or unsqueeze_motion_dim) and \
                (not (unsqueeze_pose_dim and unsqueeze_motion_dim)), \
            '2D input must unsqueeze either pose or motion dim, but not both.'

    if mean_path is not None:
        sample = unnormalize(sample, mean_path=mean_path, std_path=std_path)

    # separate sample into different smpl components, and unsqueeze axis if needed

    if unsqueeze_pose_dim:
        sample = sample.unsqueeze(1)
    if unsqueeze_motion_dim:
        sample = sample.unsqueeze(0)

    if frameskip is not None:
        # select a subset of frames to increase viz speed
        sample = sample[::frameskip]
        framerate = framerate // frameskip

    betas, thetas, trans = get_smpl_params_from_pose(sample)

    save_poses_from_smpl(betas, thetas, trans, save_path_stem=save_dir, text=text, framerate=framerate)


# visualize samples from direct dataloader output
# for consistency, try to make sure all dataloaders work with this viz
# then we can be sure all data is in the same format
def viz_from_loader(
    poses,
    cond_dict=None,
    save_dir='./',
    frameskip=1,
    device=None,
    mean_path=None,
    std_path=None,
    unsqueeze_pose_dim=False,
    unsqueeze_motion_dim=False,
    max_viz=100,
):
    if cond_dict is None:
        cond_dict = {}
    if device is None:
        device = 'cuda'
    poses = poses.to(device)
    for batch_idx in range(poses.shape[0]):

        if batch_idx >= max_viz:
            break

        pose_sample = poses[batch_idx]

        # remove padding poses
        num_frames = cond_dict['lengths'][batch_idx] if 'lengths' in cond_dict.keys() else None
        num_poses = cond_dict['num_poses'][batch_idx] if 'num_poses' in cond_dict.keys() else None
        pose_sample = remove_pad_poses(pose_sample, num_frames=num_frames, num_poses=num_poses)

        text = cond_dict['text'][batch_idx] if 'text' in cond_dict.keys() else None

        viz_pose_sample(
            pose_sample,
            save_dir=save_dir,
            frameskip=frameskip,
            text=text,
            mean_path=mean_path,
            std_path=std_path,
            unsqueeze_pose_dim=unsqueeze_pose_dim,
            unsqueeze_motion_dim=unsqueeze_motion_dim,
        )


def viz_image(image_np, caption, out_path, viz_height=30.0, title_size=None):
    # strip $ character from text for matplotlib compatibility
    # https://stackoverflow.com/questions/65655833/getting-parse-error-in-plotting-expected-end-of-text-found
    caption = caption.replace("$", "")

    # add caption title and visualize
    aspect_ratio = image_np.shape[0] / image_np.shape[1]
    fig = plt.figure(figsize=(viz_height * aspect_ratio, viz_height))
    ax = plt.axes(frameon=False, xticks=[],yticks=[])
    ax.imshow(image_np)
    ax.set_title("\n".join(wrap(caption, 40)), y=0.9, fontdict = {'fontsize' : 40})
    if title_size is not None:
        plt.gca().title.set_size(title_size)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def create_visualization_pose_blender(sample_folder):
    # assume render_pose.sh file is in the same dir as this code file
    fpath = os.path.dirname(os.path.realpath(__file__))
    obj_npy_path_abs = os.path.abspath(sample_folder)
    os.system(f"cd {fpath} && bash render_multiple.sh {obj_npy_path_abs} > /dev/null 2>&1")

def create_visualization_motion_blender(sample_folder):
    # assume render_pose.sh file is in the same dir as this code file
    fpath = os.path.dirname(os.path.realpath(__file__))
    obj_npy_path_abs = os.path.abspath(sample_folder)
    os.system(f"cd {fpath} && bash render_multiple.sh {obj_npy_path_abs}")


def add_text_to_png(obj_npy_path, text, chars_per_line=100):
    png_file_path = os.path.join(os.path.dirname(obj_npy_path), 'vertices.png')
    img = imread(png_file_path)
    # text_viz = ''
    # new_line_bool = False
    # for idx, char in enumerate(text):
    #     text_viz += char
    #     if (idx + 1) % chars_per_line == 0:
    #         new_line_bool = True
    #     if char == ' ' and new_line_bool:
    #         # new line after spaces
    #         text_viz += '\n'
    #         new_line_bool = False
    viz_image(img, text, png_file_path, title_size=60)


def create_video_from_pngs_ffmpeg(path_stem, framerate):
    # assume mp4_from_pngs.sh file is in the same dir as this code file
    fpath = os.path.dirname(os.path.realpath(__file__))
    path_stem_abs = os.path.abspath(path_stem)
    os.system(f"cd {fpath} && bash mp4_from_pngs.sh {path_stem_abs} {framerate}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_sample", type=str, default="all_result")
    args = parser.parse_args()
    
    video_sample = args.video_sample # 'pose', 'in_place', 'motion
    if video_sample == 'motion':
        sample = torch.from_numpy(np.load("/nfs/USRCSEA/IVA/Datasets/trace_motion/18597755/motion/008.npy")).float()
        #sample = torch.from_numpy(np.load("/nfs/USRCSEA/IVA/Datasets/trace_motion/25249121/motion/001.npy")).float()
        
        # 63 x 3 x 158
        viz_pose_sample(
            sample,  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
            save_dir="/home/s9053168/code/multi-pose-diffusion/pose_datasets/viz_exp",
            frameskip=5,
            text=None,
            mean_path=None,
            std_path=None,
            unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
            unsqueeze_motion_dim=False,  # set to True to pose sample with shape [num_poses, pose_dim]
            framerate=30
        )
    elif video_sample == 'in_place':
        sample = torch.from_numpy(np.load("/nfs/USRCSEA/IVA/Datasets/trace_motion/18597755/motion/008.npy")).float()
        #sample = sample[:60][::10].reshape(-1, 158)
        sample = torch.from_numpy(np.load("/home/s9053168/code/multi-pose-diffusion/pose_datasets/interhuman_teaser.npy"))
        sample = sample[5]
        sample = sample[::10, 0].reshape(-1, 158)
        breakpoint()

        
        # 63 x 3 x 158
        viz_pose_sample(
            sample,  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
            save_dir="/home/s9053168/code/multi-pose-diffusion/pose_datasets/viz_exp",
            frameskip=1,
            text=None,
            mean_path=None,
            std_path=None,
            unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
            unsqueeze_motion_dim=True,  # set to True to pose sample with shape [num_poses, pose_dim]
            framerate=30
        )
    elif video_sample == 'intergen':
        sample = torch.from_numpy(np.load("/nfs/USRCSEA/IVA/Datasets/motion_baselines/intergen-1024x61x10x158.npy")).float()
        with open("/nfs/USRCSEA/IVA/Datasets/motion_baselines/intergen-1024x61x10x158.txt", "r") as f:
            texts = f.readlines()
        
        i = 180
        viz_pose_sample(
            sample[i, :, :2],  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
            save_dir="/home/s9053168/code/multi-pose-diffusion/pose_datasets/intergen",
            frameskip=5,
            text=texts[i],
            mean_path=None,
            std_path=None,
            unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
            unsqueeze_motion_dim=False,  # set to True to pose sample with shape [num_poses, pose_dim]
            framerate=30
        )
        
    elif video_sample == 'pose':
        sample = np.load("/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/multi-pose-v1/multi_pose_4_gpu/24-02-25-22-22-36/samples/multi_pose_8_gpu/24-02-26-21-15-26/samples_1024x10x158.npz")

        for i in range(0, 1024):
            #if i in [0, 31, 24, 38, 6, 80, 188, 16, 70, 175, 185, 79, 276, 350]:
            if i > 500 and sample['num_poses'][i] == 2:
                one_sample = torch.from_numpy(sample['sample'][i, 0:sample['num_poses'][i]])
                one_sample[:, -3:] -= one_sample[:, -3:].mean(axis=0)
                if i == 175:
                    one_sample[:, -3:] *= 1.2
                    one_sample[1, -3:] *= 1.2
                if i == 70:
                    one_sample = one_sample[:6]

                viz_pose_sample(
                one_sample,  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
                save_dir="/home/s9053168/code/multi-pose-diffusion/pose_datasets/pose_paper_select_2",
                frameskip=5,
                text=None,
                mean_path=None,
                std_path=None,
                unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
                unsqueeze_motion_dim=True,  # set to True to pose sample with shape [num_poses, pose_dim]
                framerate=30
                )

    elif video_sample == "buddi":
        before = "/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/out_laion_viz/sample_00032/frame_00000/vertices.npy"
        after = "/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/out_laion_viz/sample_00033/frame_00000/vertices.npy"
        before = torch.from_numpy(np.load(before))
        after = np.load(after)
        
        before = before[[6, 5, 4, 3, 1, 7, 0, 2, 8, 9]]
        after = after[[5, 4, 2, 6, 7, 1, 9, 0, 3, 8]]
        
        breakpoint()
        save_poses_from_verts(
            before[None, :, :, :],
            text="",
            framerate=1
        )
        save_poses_from_verts(
            after[None, :, :, :],
            text="",
            framerate=1
        )    
        
    elif video_sample == "baseline":
        prompts = [909, 937, 984, 192, 772, 11]
        for idx, i in enumerate(prompts):
            intergen = torch.from_numpy(np.load(f"/nfs/USRCSEA/IVA/Datasets/motion_baselines/InterGen/smpl/{i:04d}.npy"))
            commdm = torch.from_numpy(np.load(f"/nfs/USRCSEA/IVA/Datasets/motion_baselines/ComMDM/smpl/{i:04d}.npy"))
            rig = torch.from_numpy(np.load(f"/nfs/USRCSEA/IVA/Datasets/motion_baselines/RIG/smpl/{i:04d}.npy"))
            #intergen[:, :, -3:] /= 2
            
            texts = [
            "Two people are diving underwater in a swimming pool.",
            "Two basketball players are on the court. One is dribbling the ball, while the other one is trying to steal it.",
            "There are two people playing soccer on a field.", 
            "Two women are running back and forth on a dirt road.",
            "Two women are raising their hands in celebration for victory.",
            "There are two people on the floor, with one person lying and the other one kneeling next to him."
            ]
            
            intergen[:, :, -3:] -= intergen[:, :, -3:].mean(axis=0).mean(axis=0)
            commdm[:, :, -3:] -= commdm[:, :, -3:].mean(axis=0).mean(axis=0)
            rig[:, :, -3:] -= rig[:, :, -3:].mean(axis=0).mean(axis=0)
                    
            viz_pose_sample(
                    intergen[:, :2],  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
                    save_dir="/nfs/USRCSEA/IVA/Datasets/motion_baselines/result/baseline",
                    frameskip=1,
                    text=texts[idx],
                    mean_path=None,
                    std_path=None,
                    unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
                    unsqueeze_motion_dim=False,  # set to True to pose sample with shape [num_poses, pose_dim]
                    framerate=20
                )
            
            viz_pose_sample(
                    commdm[:, :2],  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
                    save_dir="/nfs/USRCSEA/IVA/Datasets/motion_baselines/result/baseline",
                    frameskip=1,
                    text=texts[idx],
                    mean_path=None,
                    std_path=None,
                    unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
                    unsqueeze_motion_dim=False,  # set to True to pose sample with shape [num_poses, pose_dim]
                    framerate=20
                )
                        
            viz_pose_sample(
                    rig[:, :2],  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
                    save_dir="/nfs/USRCSEA/IVA/Datasets/motion_baselines/result/baseline",
                    frameskip=1,
                    text=texts[idx],
                    mean_path=None,
                    std_path=None,
                    unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
                    unsqueeze_motion_dim=False,  # set to True to pose sample with shape [num_poses, pose_dim]
                    framerate=20
                )
        
    elif video_sample == "result":
        
        list1 = [4, 48]
        list2 = [15, 28, 45]
        list3 = [26, 35]
            
        res1 = np.load("/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-v1/pose2motion_2_stage_webvid_freeze_pose_new_centered_8_gpu/24-03-06-06-00-54/samples/pose2motion_2_stage_webvid_frozen_viz_new_centered_n_1_gpu/24-03-06-23-42-14/samples_600x61x10x158.npz")
        res2 = np.load("/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-v1/pose2motion_2_stage_webvid_freeze_pose_new_centered_8_gpu/24-03-06-06-00-54/samples/pose2motion_2_stage_webvid_frozen_viz_new_centered_n_1_gpu/24-03-06-23-47-02/samples_600x61x10x158.npz")
        res3 = np.load("/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-v1/pose2motion_2_stage_webvid_freeze_pose_new_centered_8_gpu/24-03-06-06-00-54/samples/pose2motion_2_stage_webvid_frozen_viz_new_centered_2_1_gpu/24-03-06-23-59-54/samples_800x61x10x158.npz")

        res1_sample = torch.from_numpy(res1['sample'])
        res2_sample = torch.from_numpy(res2['sample'])
        res3_sample = torch.from_numpy(res3['sample'])
        for r in list3:

            viz_pose_sample(
                res3_sample[r][:, :res3['num_poses'][r]],  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
                save_dir="/home/s9053168/code/multi-pose-diffusion/pose_datasets/result_final",
                frameskip=5,
                text=None,
                mean_path=None,
                std_path=None,
                unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
                unsqueeze_motion_dim=False,  # set to True to pose sample with shape [num_poses, pose_dim]
                framerate=30
            )
        
    elif video_sample == "webvid_data":
        
        example1 = "/nfs/USRCSEA/IVA/Datasets/trace_motion/1033864628/motion/002.npy"
        example2 = "/nfs/USRCSEA/IVA/Datasets/trace_motion/10316750/motion/001.npy"
        example3 = "/nfs/USRCSEA/IVA/Datasets/trace_motion/4619537/motion/001.npy"
        example4 = "/nfs/USRCSEA/IVA/Datasets/trace_motion/25249121/motion/001.npy"
        teaser = "/nfs/USRCSEA/IVA/Datasets/trace_motion/25741751/motion/003.npy"
        method = "/nfs/USRCSEA/IVA/Datasets/trace_motion/18597755/motion/008.npy"
        one_sample = np.load(method)
        noise = np.random.normal(0, 1, size=one_sample.shape)
        one_sample[:, :, 11:] += noise[:, :, 11:] * 0.5
        one_sample = torch.from_numpy(one_sample).float()
        
        viz_pose_sample(
            one_sample,  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
            save_dir="/home/s9053168/code/multi-pose-diffusion/pose_datasets/webvid_data",
            frameskip=5,
            text=None,
            mean_path=None,
            std_path=None,
            unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
            unsqueeze_motion_dim=False,  # set to True to pose sample with shape [num_poses, pose_dim]
            framerate=30
        )
        
    elif video_sample == "all_result":
        result = "/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-v1/pose2motion_2_stage_webvid_freeze_pose_new_centered_8_gpu/24-03-06-06-00-54/samples/pose2motion_2_stage_webvid_frozen_viz_new_centered_n_1_gpu/24-03-06-23-47-02/samples_600x61x10x158.npz"
        samples = np.load(result)

        #webpage = [5, 15, 23, 37, 45, 66, 101, 112, 113, 132, 151, 186, 195, 261, 267, 342, 347, 356, 515]
        webpage = [15, 23, 45, 66, 113, 186, 195, 261, 267, 347, 356, 515]
        
        texts = [
            "One person is in the process of kicking a ball, while the other two people are trying to gain control of it.",
            "A group of people are riding bicycles on a city street.",
            "A group of people are bending down to pick up trash in a grassy area.",
            "A man is kicking a soccer ball towards a group of people on the field.",
            "A group of four young women are standing on a city street at night. They raise their arms as if talking to someone in front of them.",
            "A group of three people, including a young boy and two women, are running together in a park.",
            "A group of people are seated around a table, enjoying a meal or drinks together.",
            "A group of young women are running together in a cross-country race.",
            "There are three people riding yellow kayaks down a river.",
            "There are three people bending down to working together in a field.",
            "A group of people are raising their hands performing a lively dance.",
            "There is a group of people playing soccer on a field."
            
        ]
        
        
        # with open("/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-v1/pose2motion_2_stage_webvid_freeze_pose_new_centered_8_gpu/24-03-06-06-00-54/samples/pose2motion_2_stage_webvid_frozen_viz_new_centered_n_1_gpu/24-03-06-23-47-02/samples_600x61x10x158.txt", "r") as f:
        #     texts = f.readlines()
        
        for idx, i in enumerate(webpage):
            
            one_sample = samples['sample'][i][:, :samples['num_poses'][i]]
            
            if i == 23:
                one_sample[:, :, -3:] /= 2
            
            one_sample[:, :, :] = gaussian_filter1d(one_sample[:, :, :], sigma=1, axis=0)
            one_sample[:, :, -3:] = gaussian_filter1d(one_sample[:, :, -3:], sigma=1, axis=0)
            one_sample[:, :, -3:] -= one_sample[:, :, -3:].mean(axis=0).mean(axis=0)
            
            one_sample = torch.from_numpy(one_sample).float()
            
        
            viz_pose_sample(
                one_sample,  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
                save_dir="/nfs/USRCSEA/IVA/Datasets/motion_baselines/result/selected",
                frameskip=1,
                text=texts[idx][:-1],
                mean_path=None,
                std_path=None,
                unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
                unsqueeze_motion_dim=False,  # set to True to pose sample with shape [num_poses, pose_dim]
                framerate=20
            )
            
    elif video_sample == "two_result":
        
        result = "/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-v1/pose2motion_2_stage_webvid_freeze_pose_new_centered_8_gpu/24-03-06-06-00-54/samples/pose2motion_2_stage_webvid_frozen_viz_new_centered_2_1_gpu/24-03-06-23-59-54/samples_800x61x10x158.npz"
        samples = np.load(result)

        #webpage = [25, 26, 31, 35, 37, 120, 129, 144, 341, 379, 374, 462, 514, 543, 593, 623, 625, 721, 731, 760, 767, 778]
        #webpage = [26, 129, 341, 374, 462, 767]
        webpage = [462, 767]
    
        
        texts = [
            #"Two people are diving underwater in a swimming pool.",
            #"Two basketball players are on the court. One is dribbling the ball, while the other one is trying to steal it.",
            #"There are two people playing soccer on a field.", 
            #"Two women are running back and forth on a dirt road.",
            "Two women are raising their hands in celebration for victory.",
            "There are two people on the floor, with one person lying and the other one kneeling next to him."
        ]
        
        
        #with open("/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v3/pose2motion-ablation-v1/pose2motion_2_stage_webvid_freeze_pose_new_centered_8_gpu/24-03-06-06-00-54/samples/pose2motion_2_stage_webvid_frozen_viz_new_centered_2_1_gpu/24-03-06-23-59-54/samples_800x61x10x158.txt", "r") as f:
        #    texts = f.readlines()
        
        for idx, i in enumerate(webpage):
            
            one_sample = samples['sample'][i][:, :2]
            
            one_sample[:, :, :] = gaussian_filter1d(one_sample[:, :, :], sigma=1, axis=0)
            one_sample[:, :, -3:] = gaussian_filter1d(one_sample[:, :, -3:], sigma=1, axis=0)
            one_sample[:, :, -3:] -= one_sample[:, :, -3:].mean(axis=0).mean(axis=0)
            
            one_sample = torch.from_numpy(one_sample).float()
        
            viz_pose_sample(
                one_sample,  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
                save_dir="/nfs/USRCSEA/IVA/Datasets/motion_baselines/result/2-selected",
                frameskip=1,
                text=texts[idx],
                mean_path=None,
                std_path=None,
                unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
                unsqueeze_motion_dim=False,  # set to True to pose sample with shape [num_poses, pose_dim]
                framerate=20
            )
            
    elif video_sample == "webvid_orig":
        webvid_webpage = [474, 1176, 1407, 1459, 1633, 1731, 1745, 2125, 2593, 3171, 4828, 5695]
        with open("webvid_path.txt", "r") as f:
            all_paths = f.readlines()
        for i, video in enumerate(webvid_webpage):
            print(video)
            path = all_paths[video-1][:-1]
            video_id = path.split("/")[6]
            motion_id = int(path.split("/")[8][:-4])
            with open(f"/nfs/USRCSEA/IVA/Datasets/trace_motion/{video_id}/annot.txt", "r") as f:
                all_annots = f.readlines()
                annot = all_annots[motion_id]
                start = int(annot.split(",")[2])
                end = start + 60
                for idx in range(start, end):
                    os.makedirs(f"/nfs/USRCSEA/IVA/Datasets/trace_motion/webpage/{i+1}", exist_ok=True)
                    os.system(f"mv /nfs/USRCSEA/IVA/Datasets/trace_motion/{video_id}/{video_id}_frames/{idx:08d}.jpg \
                        /nfs/USRCSEA/IVA/Datasets/trace_motion/webpage/{i+1}")
                
                # os.system(f"ffmpeg -framerate 20 -start_number {start} -i \
                #     /nfs/USRCSEA/IVA/Datasets/trace_motion/{video_id}/{video_id}_frames/%08d.jpg \
                #         -r 20 -pix_fmt yuv420p -frames:v 60 -vf 'fps=20' {video_id}.mp4")

                # if int(video_id) in [4619537, 1008700546, 1033864628]:
                #     os.system(f"ffmpeg -i {video_id}.mp4 -vf 'crop=iw:ih/2:0:0' -c:a copy {video_id}_crop.mp4")
                
    elif video_sample == "webvid_webpage":
        
        #webvid_webpage = [474, 1176, 1407, 1459, 1633, 1731, 1745, 2125, 2593, 3171, 4828, 5695]
        webvid_webpage = [474, 1176, 1407, 1459, 1633, 1731, 1745, 2125, 2593, 3171, 4828, 5695]
        
        with open("webvid_path.txt", "r") as f:
            all_paths = f.readlines()
        for i in webvid_webpage:
            path = all_paths[i-1][:-1]
            video_id = path.split("/")[6]
            motion = np.load(path)
            motion = motion[:60]
            motion[:, :, :] = gaussian_filter1d(motion[:, :, :], sigma=1, axis=0)
            motion[:, :, -3:] = gaussian_filter1d(motion[:, :, -3:], sigma=1, axis=0)
            motion[:, :, -3:] -= motion[:, :, -3:].mean(axis=0).mean(axis=0)
            motion = torch.from_numpy(motion).float()
            
            with open(f"/nfs/USRCSEA/IVA/Datasets/trace_motion/{video_id}/text.txt", "r") as f:
                text = f.readlines()[0]
            
            viz_pose_sample(
                motion,  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
                save_dir="/nfs/USRCSEA/IVA/Datasets/motion_baselines/webvid_webpage",
                frameskip=1,
                text=text,
                mean_path=None,
                std_path=None,
                unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
                unsqueeze_motion_dim=False,  # set to True to pose sample with shape [num_poses, pose_dim]
                framerate=20
            )
            
    elif video_sample == "ablation":
        with open("/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mengyi-v3/ablation/ablation_0_0_8_gpu/24-03-13-01-26-48/samples_1024x61x10x158.txt", "r") as f:
            texts = f.readlines()
            
        from glob import glob
        
        for i in range(1000):
            for p in [0, 4]:
                for m in [0, 2, 4, 6]:
                    sample_file = glob(f"/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mengyi-v3/ablation/ablation_{p}_{m}_8_gpu/*/samples_1024x61x10x158.npz")[0]
                    sample = np.load(sample_file)
                    
                    one_sample = sample['sample'][i][:, :sample['num_poses'][i]]

                    #one_sample[:, :, :] = gaussian_filter1d(one_sample[:, :, :], sigma=1, axis=0)
                    #one_sample[:, :, -3:] = gaussian_filter1d(one_sample[:, :, -3:], sigma=1, axis=0)
                    one_sample[:, :, -3:] -= one_sample[:, :, -3:].mean(axis=0).mean(axis=0)
                    
                    one_sample = torch.from_numpy(one_sample).float()
                    
                    viz_pose_sample(
                        one_sample,  # input shape can be: [seq_len, pose_dim], [num_poses, pose_dim], or [seq_len, num_poses, pose_dim]
                        save_dir=f"/nfs/USRCSEA/IVA/Datasets/motion_baselines/result/ablation-viz/{i}",
                        frameskip=10,
                        text=texts[i][:-1],
                        mean_path=None,
                        std_path=None,
                        unsqueeze_pose_dim=False,  # set to True for motion sample with shape [seq_len, pose_dim]
                        unsqueeze_motion_dim=False,  # set to True to pose sample with shape [num_poses, pose_dim]
                        framerate=20
                    )