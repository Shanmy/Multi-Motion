import math
import random

from PIL import Image
from decord import VideoReader, cpu
import numpy as np


def prepare_pil_image(pil_image, resolution, random_crop=False, random_flip=False, bilinear=False):
    if random_crop:
        # TODO: double-check non-square random crop
        assert resolution[0] == resolution[1], 'Random crop only verified for square images currently.'
        arr = _random_crop_arr(pil_image, resolution[0], bilinear=bilinear)
    else:
        arr = _center_crop_arr(pil_image, resolution, bilinear=bilinear)
    if random_flip and random.random() < 0.5:
        arr = arr[:, ::-1]
    arr = arr.astype(np.float32) / 127.5 - 1
    return arr


def prepare_video_frames(
    path,
    resolution,
    frames,
    frameskip,
    resize=False,
    random_crop=False,
    random_flip=False,
    bilinear=False,
):
    # TODO: improve to make more efficient?

    # open video
    video_reader = VideoReader(path, ctx=cpu(0))

    #fps_orig = video_reader.get_avg_fps()
    frames_partial = list(range(0, len(video_reader), frameskip))
    start_idx = random.randint(0, max(0, len(frames_partial) - frames))
    frame_indices = frames_partial[start_idx:start_idx+min(frames, len(frames_partial))]
    vid_frames = video_reader.get_batch(frame_indices)
    vid_frames = vid_frames.asnumpy()

    if vid_frames.shape[0] < frames:
        vid_frames = _mirror_pad(vid_frames, frames)

    if resize:
        arr = np.zeros([
            frames,
            resolution[0],
            resolution[1],
            vid_frames.shape[3],
        ]).astype(np.float32)
        for i in range(frames):
            # NOTE: for now, random crop must be False. TODO: add code to crop all vid frames in same way?
            assert not random_crop
            arr[i] = prepare_pil_image(
                            Image.fromarray(vid_frames[i]),
                            resolution,
                            random_crop=False,
                            random_flip=False,
                            bilinear=bilinear,
                        )
    else:
        arr = vid_frames.astype(np.float32) / 127.5 - 1

    if random_flip and random.random() < 0.5:
        # random flip of all video frames together
        arr = arr[:, :, ::-1]

    return arr


def _mirror_pad(vid, frames):
    for _ in range(frames):
        if vid.shape[0] < frames:
            vid = np.concatenate((vid, vid[::-1]), 0)
        else:
            break

    # backup zero pad (should not be used if code is working)
    if vid.shape[0] < frames:
        print('Warning: Backup zero pad used.')
        num_missing_frames = frames - vid.shape[0]
        zero_vid = np.zeros([num_missing_frames] + list(vid.shape[1:]))
        vid = np.concatenate((vid, zero_vid), 0)

    return vid[0:frames]


def _center_crop_arr(pil_image, image_size, bilinear=False):

    while (pil_image.size[1] >= 2 * image_size[0]) and (pil_image.size[0] >= 2 * image_size[1]):
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = max(image_size[0] / pil_image.size[1], image_size[1] / pil_image.size[0])
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=(Image.BILINEAR if bilinear else Image.BICUBIC),
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size[0]) // 2
    crop_x = (arr.shape[1] - image_size[1]) // 2
    return arr[crop_y : crop_y + image_size[0], crop_x : crop_x + image_size[1]]


def _random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0, bilinear=False):
    # square-size random crop only
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), 
        resample=(Image.BILINEAR if bilinear else Image.BICUBIC),
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
