import os
from datetime import datetime

import numpy as np
import torch

import chamfer_distance as chd
from trimesh import Trimesh

from smpl.smpl import SMPLA_parser
from pose_datasets.data_utils import get_smpl_params_from_pose, convert_smpl_to_6d, remove_pad_poses
from pose_datasets.laion_pose_loader import LaionPose
from viz.viz_utils import save_poses_from_verts


def calc_sdf(vertices, faces):
    verts_x, verts_y = vertices[0], vertices[1]

    # calculate vertex normals
    # need to detach gradients for Trimesh
    mesh_x = Trimesh(vertices=verts_x.clone().detach().cpu().numpy(), faces=faces.cpu())
    mesh_y = Trimesh(vertices=verts_y.clone().detach().cpu().numpy(), faces=faces.cpu())
    normals_x = torch.from_numpy(mesh_x.vertex_normals.astype(np.float32)).to(vertices.device)
    normals_y = torch.from_numpy(mesh_y.vertex_normals.astype(np.float32)).to(vertices.device)

    # attach dimension for batch size to use point2point_signed
    verts_x, verts_y = verts_x[None, :, :], verts_y[None, :, :]
    normals_x, normals_y = normals_x[None, :, :], normals_y[None, :, :]

    # calculate the sdfs for mesh_x and mesh_y
    y2x_signed, x2y_signed, _, _ = \
        point2point_signed(verts_x, verts_y, x_normals=normals_x, y_normals=normals_y)

    # the sign distance from mesh a to b, get mask where everything < 0.0 is set to True
    y2x_mask = y2x_signed < 0.0
    y2x_signed = y2x_mask * y2x_signed

    return torch.sum(y2x_signed)


def get_pair_list(num_pairs):
    pair_list = []
    for i in range(num_pairs - 1):
        pairs_i = []
        for j in range(i + 1, num_pairs):
            pairs_i.append([i, j])
        pair_list.append(pairs_i)
    return pair_list


def calc_pairwise_sdf(vertices, faces):
    # calculate pairwise sdf for all pose pairs inside
    pairs = get_pair_list(vertices.shape[0])
    total_sdf = 0

    for pair_list in pairs:
        for pose_pair in pair_list:
            pair_vertices = vertices[pose_pair]
            sum_sdf = calc_sdf(pair_vertices, faces)

            if abs(sum_sdf) == 0.0: # if the loss is zero then continue
                continue

            total_sdf += sum_sdf

    if abs(total_sdf) == 0:
        return 0

    return total_sdf


def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
        return_vector=False,
):
    """
    signed distance between two pointclouds
    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
    Returns:
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - x2y_signed: Torch.Tensor
            the sign distance from x to y
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y
    """

    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x, y, x_normals=x_normals, y_normals=y_normals)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near  # y point to x
    y2x = y - y_near  # x point to y

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    if not return_vector:
        return y2x_signed, x2y_signed, yidx_near, xidx_near
    else:
        return y2x_signed, x2y_signed, yidx_near, xidx_near, y2x, x2y


# update thetas and trans parameters so that mesh does not intersect
def gradient_descent_loop(
    poses,
    cond_dict,
    learning_rate_thetas=0.0005,
    learning_rate_trans=0.0005,
    max_iters=10,
    save_dir=None,
    smpla_parser=None,
    print_log=True
):

    assert len(poses.shape) == 3  # [num_frames, num_poses, pose_dim]
    assert poses.shape[-1] == 158, 'Currently only support 158 dim pose data.'

    # load smpl models
    if smpla_parser is None:
        smpla_parser = SMPLA_parser().to(poses.device)

    for pose_idx, (pose, num_poses) in enumerate(zip(poses, cond_dict['num_poses'])):

        pose_nonpadded = remove_pad_poses(pose, num_frames=None, num_poses=num_poses)
        betas, thetas, trans = get_smpl_params_from_pose(pose_nonpadded)

        thetas = torch.autograd.Variable(thetas.clone(), requires_grad=True)
        trans = torch.autograd.Variable(trans.clone(), requires_grad=True)

        verts_list = []

        step = 0 # number of iterations run
        while True:

            # loss to minimize
            vertices, faces = smpla_parser.forward_trans(betas, thetas, trans, center_trans=False)
            average_neg_sdf = - calc_pairwise_sdf(vertices, faces)
            if abs(average_neg_sdf) == 0: # if we have no more intersections
                break

            # calculate gradients = backward pass
            thetas_grad, trans_grad = torch.autograd.grad(average_neg_sdf, [thetas, trans])

            # update vars
            with torch.no_grad():
                thetas.data -= learning_rate_thetas * thetas_grad
                trans.data -= learning_rate_trans * trans_grad

            verts_list.append(vertices.clone().detach().cpu())

            if print_log:
                # printout
                print("step:", step, "average_neg_sdf:", average_neg_sdf)

            step += 1
            if step >= max_iters:
                break

        if save_dir is not None:
            # viz current mesh results
            save_poses_from_verts(torch.stack(verts_list), save_path_stem=save_dir)

        # update smpl parameters
        thetas = convert_smpl_to_6d(thetas)
        poses[pose_idx, :num_poses] = torch.cat((betas, thetas.detach(), trans.detach()), dim=1)

    return poses


# get samples with overlap from dataloader (for demo only)
def get_poses_to_shift(batch_size=1, smpla_parser=None):

    # load smpl models
    if smpla_parser is None:
        smpla_parser = SMPLA_parser().to('cuda')

    # make dataloader and get batch
    pose_dataloader = LaionPose(
        normalize=False,
        batch_size=batch_size,
        resampled=True,
        concat_vars=True,
        yield_cond_dict=True,
    )
    poses, cond_dict = next(iter(pose_dataloader))
    poses = poses.cuda()

    poses_out = []
    num_poses_out = []

    for sample_idx, (pose, num_poses) in enumerate(zip(poses, cond_dict['num_poses'])):

        pose_nonpadded = remove_pad_poses(pose, num_frames=None, num_poses=num_poses)
        betas, thetas, trans = get_smpl_params_from_pose(pose_nonpadded)

        if num_poses >  1:
            # instantiate converter and get sdf
            vertices, faces = smpla_parser.forward_trans(betas, thetas, trans)
            sum_sdf = calc_pairwise_sdf(vertices, faces)
            print("SDF sum: ", sum_sdf)

            # get the thetas with larget negative sdfs to use for debugging if there is an intersection
            if sum_sdf < -5.0:
                poses_out.append(pose)
                num_poses_out.append(num_poses)

    if not poses_out == []:
        poses_out = torch.stack(poses_out)
    cond_dict_out = {"num_poses": num_poses_out}  # only need number poses in dict for this use case
    return poses_out, cond_dict_out


if __name__ == "__main__":

    batch_size = 50
    learning_rate_thetas = 0.0001
    learning_rate_trans = 0.0001
    max_iters = 5

    # visualize from dataloader

    save_dir = 'out_corrector/overlapping_poses'
    save_dir = os.path.join(save_dir, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    os.makedirs(save_dir)

    # save pose to disk for processing
    poses, cond_dict = get_poses_to_shift(batch_size=batch_size)

    if not poses == []:
        # process saved poses
        gradient_descent_loop(
            poses,
            cond_dict,
            learning_rate_thetas=learning_rate_thetas,
            learning_rate_trans=learning_rate_trans,
            max_iters=max_iters,
            save_dir=save_dir
        )
