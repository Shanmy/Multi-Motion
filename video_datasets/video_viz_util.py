import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch


def convert_to_grid(images, H_grid=None, W_grid=None, sep_width=1):
    B, H, W, C = images.shape
    if H_grid is None:
        H_grid = int(np.ceil(B ** 0.5))
    if W_grid is None:
        W_grid = int(np.ceil(B / H_grid))
    assert B <= H_grid * W_grid, 'Batch size must be less than or equal to the number of grid slots.'

    H_grid_pix = H * H_grid + (H_grid - 1) * sep_width
    W_grid_pix = W * W_grid + (W_grid - 1) * sep_width
    image_grid = np.ones([H_grid_pix, W_grid_pix, C]).astype(images.dtype)

    for i in range(images.shape[0]):
        image = images[i]

        H_grid_loc = i // W_grid
        W_grid_loc = i % W_grid

        H_start = H_grid_loc * (H + sep_width)
        H_end = H_start + H
        W_start = W_grid_loc * (W + sep_width)
        W_end = W_start + W

        image_grid[H_start:H_end, W_start:W_end] = image

    return image_grid


def viz_states(path, states, tag='out', save_vid_viz=True, fps=8, save_viz_grid=True):
    # viz tensor pixel range should be either [0, 255] in uint8 format or [-1.0, 1.0] in float format
    viz_ims(path, states, tag)
    if len(states.shape) == 5 and save_vid_viz:
        if save_viz_grid:
            viz_vids_grid(path, states, tag, fps=fps)
        else:
            viz_vids_single(path, states, tag, fps=fps)


def viz_ims(path, ims, tag='out'): 
    if torch.is_tensor(ims):
        ims = ims.numpy()

    if len(ims.shape) == 5 and ims.shape[2] > 1:
        video_mode = True
        H_grid = ims.shape[0]
        W_grid = ims.shape[2]
        ims = ims.transpose(0, 2, 3, 4, 1)
        ims = ims.reshape((-1,) + ims.shape[2:5])
    elif len(ims.shape) == 5 and ims.shape[2] == 1:
        # single-frame videos from treated as images
        video_mode = False
        H_grid = None
        W_grid = None
        ims = ims.squeeze(2).transpose(0, 2, 3, 1)
    else:
        video_mode = False
        H_grid = None
        W_grid = None
        ims = ims.transpose(0, 2, 3, 1)

    if not ims.dtype == np.uint8:
        ims = (np.clip(ims, -1., 1.) + 1) / 2

    image_grid = convert_to_grid(ims, H_grid=H_grid, W_grid=W_grid)

    out_file = os.path.join(path, f'samples_{tag}.png')
    plt.imsave(out_file, image_grid, format="png", dpi=2000)


# make a video grid that shows all video samples in tensor at the same time
def viz_vids_grid(path, vids, tag='out', fps=8):
    if torch.is_tensor(vids):
        vids = vids.numpy()

    vids = vids.transpose(0, 2, 3, 4, 1)
    if not vids.dtype == np.uint8:
        vids = 255 * (np.clip(vids, -1., 1.) + 1) / 2
        vids = vids.astype(np.uint8)

    vid_grid_test = convert_to_grid(vids[:, 0])
    height = vid_grid_test.shape[0]
    width = vid_grid_test.shape[1]

    out_file = os.path.join(path, f'vids_{tag}.mp4')
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), True)
    for i in range(vids.shape[1]):
        vids_frame = vids[:, i]
        vids_frame_grid = convert_to_grid(vids_frame)
        out.write(cv2.cvtColor(vids_frame_grid, cv2.COLOR_RGB2BGR))
    out.release()


# make individual videos of each video sample in tensor
def viz_vids_single(path, vids, tag='out', fps=8):
    if torch.is_tensor(vids):
        vids = vids.numpy()

    vids = vids.transpose(0, 2, 3, 4, 1)
    if not vids.dtype == np.uint8:
        # float-valued vids expected to have pixel values in range [-1, 1]
        vids = 255 * (np.clip(vids, -1., 1.) + 1) / 2
        vids = vids.astype(np.uint8)

    height = vids.shape[2]
    width = vids.shape[3]

    for i in range(vids.shape[0]):
        vid = vids[i]
        out_file = os.path.join(path, f'vid_{i}_{tag}.mp4')
        out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), True)
        for j in range(vid.shape[0]):
            out.write(cv2.cvtColor(vid[j], cv2.COLOR_RGB2BGR))
        out.release()
