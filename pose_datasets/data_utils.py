import os
from itertools import repeat
import random

import numpy as np

import torch
from torch.utils.data import DataLoader

from rotation_conversions import (
    axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle
)


def truncate_frames(pose, max_len):
    if pose.shape[0] > max_len:
        start_frame = random.randint(0, pose.shape[0] - max_len - 1)
        pose_trunc = pose[start_frame:(start_frame + max_len)]
    else:
        pose_trunc = pose
    return pose_trunc


def make_shorter_instruct_blip_text(text, sentence_probs=(0.45, 0.45, 0.1)):
    text = text.lower()
    # remove recurring uninformative strings from instruct blip
    text = text.replace('in the image, ', '')
    text = text.replace('in the image ', '')
    # take only the few sentences
    text = text.split('.')
    text_out = ''

    num_sentences = int(np.random.choice(list(np.arange(len(sentence_probs)) + 1), 1, p=sentence_probs))
    num_sentences = min(num_sentences, len(text))

    for idx in range(num_sentences):
        text_out += text[idx]
        text_out += '.'
    return text_out


def list_files_recursively(data_dir, file_types=("jpg", "jpeg", "png", "gif")):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if file_types is None or "." in entry and ext.lower() in file_types:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(list_files_recursively(full_path, file_types=file_types))
    return results


def process_translations_from_padded_poses(
    poses,
    num_frames=None,
    num_poses=None,
    pose_view=False,
    trans_aug_rad=None,
    single_motion_mode=False,
    center_first_frame=True,
    center_trans=True,
    return_shift=False,
    process_height=True,
):
    for idx in range(poses.shape[0]):

        # get translations for non-padded poses
        if len(poses.shape) == 4:
            trans_idx = poses[idx, :num_frames[idx], :num_poses[idx], -3:]
        elif num_poses is None:
            trans_idx = poses[idx, :num_frames[idx], -3:]
        else:
            trans_idx = poses[idx, :num_poses[idx], -3:]

        if len(trans_idx.shape) == 3 and pose_view:
            # reshape all poses to 2D tensor (frames on batch dim)
            trans_idx = trans_idx.contiguous().view(-1, 3)
            reshape_out = True
        elif len(trans_idx.shape) == 3 and not pose_view:
            # reshape all poses to 2D tensor (poses on batch dim)
            trans_idx = trans_idx.permute(1, 0, 2).contiguous().view(-1, 3)
            reshape_out = True
        else:
            reshape_out = False

        # center translations (either do it for each from, or center according to first frame for "single_motion_mode")
        trans_idx = process_translations(
                            trans_idx.cpu().numpy(),
                            trans_aug_rad=trans_aug_rad,
                            single_motion_mode=single_motion_mode,
                            center_first_frame=center_first_frame,
                            center_trans=center_trans,
                            return_shift=return_shift,
                            process_height=process_height,
                        )
        trans_idx = torch.tensor(trans_idx).to(poses.device)

        if reshape_out and pose_view:
            trans_idx = trans_idx.view(-1, num_poses[idx], 3).contiguous()
        if reshape_out and not pose_view:
            trans_idx = trans_idx.view(-1, num_frames[idx], 3).permute(1, 0, 2).contiguous()

        if len(poses.shape) == 4:
            poses[idx, :num_frames[idx], :num_poses[idx], -3:] = trans_idx
        elif num_poses is None:
            poses[idx, :num_frames[idx], -3:] = trans_idx
        else:
            poses[idx, :num_poses[idx], -3:] = trans_idx

    return poses


def process_translations(
    trans,
    trans_aug_rad=None,
    single_motion_mode=False,
    center_first_frame=True,
    center_trans=True,
    return_shift=False,
    process_height=True,
):

    total_shift = np.zeros_like(trans)

    reshape_out = False
    if len(trans.shape) == 3:
        reshape_out = True
        num_frames = trans.shape[0]
        num_poses = trans.shape[1]
        if center_trans and center_first_frame:
            # origin is center of all people along first frame
            mean_shift = - trans[0].mean(axis=0, keepdims=True)
        trans = trans.reshape(-1, trans.shape[2])
        total_shift = total_shift.reshape(-1, total_shift.shape[2])
        if center_trans and not center_first_frame:
            # origin is center of all people along all frames
            mean_shift = - trans.mean(axis=0, keepdims=True)
        if not center_trans:
            # origin from the original data space
            mean_shift = np.zeros_like(trans)
    elif center_trans and center_first_frame and single_motion_mode:
        # origin is given by first frame translation
        mean_shift = - trans[:1]
    elif center_trans:
        # origin is center of all frames
        mean_shift = - trans.mean(axis=0, keepdims=True)
    else:
        # origin from original data space
        mean_shift = np.zeros_like(trans)
    if not process_height:
        mean_shift[..., -1] = 0

    trans += mean_shift
    total_shift += mean_shift

    if trans_aug_rad is not None:
        rand_shift = (np.array(trans_aug_rad) * np.random.normal(size=[3])).reshape(1, 3)
        if not process_height:
            rand_shift[..., -1] = 0
        trans += rand_shift
        total_shift += rand_shift

    if reshape_out:
        trans = trans.reshape(num_frames, num_poses, -1)
        total_shift = total_shift.reshape(num_frames, num_poses, -1)

    if return_shift:
        return trans, total_shift
    else:
        return trans


def pose_random_rotation(pose):

    assert pose.shape[-1] == 158

    if len(pose.shape) == 3:
        reshape_out = True
        num_frames = pose.shape[0]
        num_poses = pose.shape[1]
        pose = pose.reshape(num_frames * num_poses, -1)
    else:
        reshape_out = False

    # random rotation around z axis
    angle_radians = np.radians(360 * np.random.uniform())
    angle_radians_neg = (2 * np.pi) - angle_radians
    rotation_matrix = torch.Tensor([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])
    rotation_matrix_neg = torch.Tensor([
        [np.cos(angle_radians_neg), -np.sin(angle_radians_neg), 0],
        [np.sin(angle_radians_neg), np.cos(angle_radians_neg), 0],
        [0, 0, 1]
    ])

    # rotate global translations
    global_trans = torch.from_numpy(pose[..., -3:])
    new_global_trans = torch.matmul(global_trans, rotation_matrix).numpy()
    pose[..., -3:] = new_global_trans

    # global orientations rotate in opposite dir of translations
    global_rot = rotation_6d_to_matrix(torch.from_numpy(pose[..., 11:17]))
    new_global_rot = matrix_to_rotation_6d(torch.matmul(rotation_matrix_neg, global_rot)).numpy()
    pose[..., 11:17] = new_global_rot

    if reshape_out:
        pose = pose.reshape(num_frames, num_poses, -1)

    return pose


def get_valid_pose_sample_count(lengths=None, num_poses=None, device=None):
    assert not (lengths is None and num_poses is None)
    if device is None:
        device = 'cuda'
    batch_size = (len(lengths) if lengths is not None else len(num_poses))
    valid_pose_sample_count = torch.ones(size=[batch_size]).to(device)
    if lengths is not None:
        valid_pose_sample_count =  valid_pose_sample_count * lengths.to(device)
    if num_poses is not None:
        valid_pose_sample_count = valid_pose_sample_count * num_poses.to(device)
    return valid_pose_sample_count


def get_pose_pad_batch(poses, lengths=None, num_poses=None, max_len=None, max_poses=None):
    if num_poses is None:
        # for 1D model with outputs along pose/frame dim, lengths/max_len args and set num_poses/max_poses to None
        assert lengths is not None and max_len is not None
        pad_pose = torch.zeros([poses.shape[0], max_len, poses.shape[-1]]).to(poses.device)
        for i in range(poses.shape[0]):
            pad_pose_i = get_pose_pad(poses[i, :lengths[i]], max_len=max_len)
            pad_pose[i] = pad_pose_i
    else:
        # pad along both frame and pose dim
        pad_pose = torch.zeros([poses.shape[0], max_len, max_poses, poses.shape[-1]]).to(poses.device)
        for i in range(poses.shape[0]):
            pad_pose_i = get_pose_pad(
                poses[i, :lengths[i], :num_poses[i]], max_len=max_len, max_poses=max_poses
            )
            pad_pose[i] = pad_pose_i
    return pad_pose


# add dummy poses to end of frame/pose sequence so all sequences have same length
def get_pose_pad(poses, max_len=None, max_poses=None, return_npy=False):
    if not torch.is_tensor(poses):
        poses = torch.tensor(poses).float()
    if max_poses is None:
        # for 1D padding along pose/frame dim, use max_len arg and set max_pose to None
        assert max_len is not None
        length = min(max_len, poses.shape[0])
        poses_pad = torch.zeros([max_len, poses.shape[-1]]).to(poses.device)
        poses_pad[:length] = poses[:length]
    else:
        length = min(max_len, poses.shape[0])
        num_poses = min(max_poses, poses.shape[1])
        poses_pad = torch.zeros([max_len, max_poses, poses.shape[-1]]).to(poses.device)
        poses_pad[:length, :num_poses] = poses[:length, :num_poses]
    if return_npy:
        poses_pad = poses_pad.cpu().numpy()
    return poses_pad


def get_masks_from_size(
    seq_lens,
    max_len,
    device=None,
):

    masks = []

    # add new mask keys for both motion and pose. TODO: do this batchwise instead of in a for loop?
    for batch_idx in range(len(seq_lens)):

        mask = torch.ones([max_len]).float()
        # TODO: mask before and after instead of just after?
        mask[:min(max_len, int(seq_lens[batch_idx]))] = 0  # unmasked (observed) states.
        masks.append(mask)

    masks = torch.stack(masks)

    if device is not None:
        masks = masks.to(device)

    return masks


def set_dummy_poses_to_zero(poses, lengths=None, num_poses=None):
    if lengths is not None:
        motion_mask = get_masks_from_size(
            seq_lens=lengths, max_len=poses.shape[1], device=poses.device
        )
        if len(poses.shape) == 4:
            motion_mask = motion_mask.unsqueeze(-1).unsqueeze(-1)
        else:
            motion_mask = motion_mask.unsqueeze(-1)
        poses = poses * (1 - motion_mask)
    if num_poses is not None:
        poses_max_len = (poses.shape[2] if len(poses.shape) == 4 else poses.shape[1])
        pose_mask = get_masks_from_size(
            seq_lens=num_poses, max_len=poses_max_len, device=poses.device
        )
        if len(poses.shape) == 4:
            pose_mask = pose_mask.unsqueeze(1).unsqueeze(-1)
        else:
            pose_mask = pose_mask.unsqueeze(-1)
        poses = poses * (1 - pose_mask)
    return poses


def make_dataloader(dataset, batch_size, shuffle=True, num_workers=8, drop_last=True):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    dataloader = repeater(dataloader)
    return dataloader


def repeater(dataloader):
    for loader in repeat(dataloader):
        for data in loader:
            yield data


def unnormalize(pose, mean_path, std_path):
    mean = np.load(mean_path)
    std = np.load(std_path)

    assert pose.shape[-1] == 158, "wrong pose feature dimension"
    mean = mean.reshape([1] * (len(pose.shape) - 1) + [-1])
    std = std.reshape([1] * (len(pose.shape) - 1) + [-1])

    mean = torch.tensor(mean).float().to(pose.device)
    std = torch.tensor(std).float().to(pose.device)

    pose = pose * std + mean

    return pose


def get_pose_from_smpl_params(
    betas,
    thetas,
    trans,
    process_trans=True,
    trans_aug_rad=None,
    convert_6d=False,
    align_pose=False,
    return_shift=False,
):
    if convert_6d:
        # convert smpl thetas to 6d thetas
        thetas = convert_smpl_to_6d(thetas).cpu().numpy()

    if align_pose:
        # align the global coordinates of bev outputs with the amass global coordinates
        thetas_and_trans = np.concatenate((thetas, trans), 1)
        thetas_and_trans = align_laion_pose(thetas_and_trans)
        thetas = thetas_and_trans[..., :-3]
        trans = thetas_and_trans[..., -3:]

    # center and augment the translations
    if process_trans:
        trans, trans_shift = process_translations(trans, trans_aug_rad=trans_aug_rad, return_shift=True)
    else:
        trans_shift = np.zeros_like(trans)

    # concat order is [beta (11D), theta (144D after 6d conversion), trans (3D)]
    pose = np.concatenate((betas, thetas, trans), 1)

    if return_shift:
        return pose, trans_shift
    else:
        return pose


def get_smpl_params_from_pose(
    pose,
    reverse_align_pose=False,
    trans_shift=None,
):

    assert pose.shape[-1] == 158

    betas = pose[..., :11]
    thetas = pose[..., 11:155]
    trans = pose[..., 155:]

    if trans_shift is not None:
        trans -= trans_shift

    if reverse_align_pose:
        thetas_and_trans = torch.cat((thetas, trans), -1)
        thetas_and_trans = reverse_align_laion_pose(thetas_and_trans)
        thetas = thetas_and_trans[..., :-3]
        trans = thetas_and_trans[..., -3:]

    thetas = convert_6d_to_smpl(thetas)

    return betas, thetas, trans


def remove_pad_poses(pose, num_frames, num_poses):
    if num_frames is not None:
        pose = pose[:num_frames]
    if num_poses is not None:
        if len(pose.shape) == 2:
            # pose idx comes first
            pose = pose[:num_poses]
        else:
            # pose idx comes second
            pose = pose[:, :num_poses]
    return pose


def convert_smpl_to_6d(thetas_smpl):
    if isinstance(thetas_smpl, np.ndarray):
        thetas_smpl = torch.from_numpy(thetas_smpl)
    thetas_6d = matrix_to_rotation_6d(axis_angle_to_matrix(thetas_smpl.reshape(-1, 24, 3)))
    if len(thetas_smpl.shape) == 2:
        thetas_6d = thetas_6d.reshape(thetas_smpl.shape[0], 144)
    else:
        thetas_6d = thetas_6d.reshape(thetas_smpl.shape[0], thetas_smpl.shape[1], 144)
    return thetas_6d


def convert_6d_to_smpl(thetas_6d):
    thetas_smpl = matrix_to_axis_angle(rotation_6d_to_matrix(thetas_6d.reshape(-1, 24, 6)))
    if len(thetas_6d.shape) == 2:
        thetas_smpl = thetas_smpl.reshape(thetas_6d.shape[0], 72)
    else:
        thetas_smpl = thetas_smpl.reshape(thetas_6d.shape[0], thetas_6d.shape[1], 72)
    return thetas_smpl


def align_laion_pose(pose, angle=270, shift_trans=False, shift_vec=(0, 0, 0)):

    assert pose.shape[-1] == 147, 'Input should be 147 vector, 144 dims for 6d thetas and 3 for global translation'

    angle_radians = np.radians(angle)

    rotation_matrix_x = torch.Tensor([
        [1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)]
    ])  # x

    rotation_matrix_z = torch.Tensor([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])  # z

    rotation_matrix = torch.matmul(rotation_matrix_z, rotation_matrix_x)

    global_rot = rotation_6d_to_matrix(torch.from_numpy(pose[..., :6])) # [... x 6]
    new_global_rot = matrix_to_rotation_6d(torch.matmul(rotation_matrix, global_rot)).numpy()
    pose[..., :6] = new_global_rot

    trans = np.copy(pose[..., -3:])
    pose[..., -3] = trans[..., -1]  # z -> x
    pose[..., -2] = -trans[..., -3]  # x -> y
    pose[..., -1] = -trans[..., -2]  # y -> z

    if shift_trans:
        # make laion/amass roughly match, FIXME later?
        pose[..., -3] += shift_vec[0]
        pose[..., -2] += shift_vec[1]
        pose[..., -1] += shift_vec[2]

    return pose


def reverse_align_laion_pose(pose, angle=90, shift_trans=False, shift_vec=(0, 0, 0)):

    assert pose.shape[-1] == 147, 'Input should be 147 vector, 144 dims for 6d thetas and 3 for global translation'

    angle_radians = np.radians(angle)

    rotation_matrix_x = torch.Tensor([
        [1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)]
    ])  # x

    rotation_matrix_z = torch.Tensor([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])  # z

    rotation_matrix = torch.matmul(rotation_matrix_x, rotation_matrix_z).to(pose.device)

    global_rot = rotation_6d_to_matrix(pose[..., :6]) # [... x 6]
    new_global_rot = matrix_to_rotation_6d(torch.matmul(rotation_matrix.double(), global_rot.double()).double())
    pose[..., :6] = new_global_rot

    if shift_trans:
        pose[..., -3] -= shift_vec[0]
        pose[..., -2] -= shift_vec[1]
        pose[..., -1] -= shift_vec[2]

    trans = pose[..., -3:].clone()
    pose[..., -1] = trans[..., -3]  # z -> x
    pose[..., -3] = -trans[..., -2]  # x -> y
    pose[..., -2] = -trans[..., -1]  # y -> z

    return pose
