# TODO: update this file?

import os
import argparse
from datetime import datetime
import tarfile
import json

from PIL import Image

import torch
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
)

import webdataset as wds

from data_laion import Laion
from net_utils import import_bev, import_clip
from processing_utils import clip_filter
from viz_utils import viz_image

import sys
sys.path.append('./open_clip/src/')
from open_clip.factory import create_model_and_transforms
sys.path.append('../pose_datasets/')
from clip_tokenizer import tokenize as clip_tokenizer


def apply_clip_filter(
    image_clip,
    clip_model,
    filter_phrases,
    filter_phrase_thresholds,
    clip_type,
):

    pass_filter, closest_filter_phrase = clip_filter(
        image_clip=image_clip.cuda(),
        clip_model=clip_model,
        filter_phrases=filter_phrases,
        filter_phrase_thresholds=filter_phrase_thresholds,
        clip_tokenizer=clip_tokenizer,
        clip_type=clip_type,
    )

    return pass_filter, closest_filter_phrase


def get_clip_model(clip_type, clip_ckpt, device=torch.device(0)):
    if clip_type == 'orig':
        clip_model = import_clip(device)
    elif clip_type == 'open_clip':
        clip_model, _, _ = create_model_and_transforms(
            'ViT-H-14',
            '/nfs/USRCSEA/IVA/Models/PoseCaptionData/open_clip/vit_h_14_open_clip_pytorch_model.bin',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=None,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            aug_cfg={},
            output_dict=True,
        )
    else:
        raise ValueError('Invalid "clip_type".')
    return clip_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_type", type=str, default="orig", help="'open_clip' or 'orig'")
    parser.add_argument("--laion_tar_files", type=str, default="/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data4/{00000..32449}.tar")
    parser.add_argument("--out_path", type=str, default="out_laion_viz/")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--burnin_batches", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument(
        "--filter_phrases",
        type=str,
        #default="",
        default="a webpage screen with text,a poster with text,a picture of a piece of clothing,a picture of a shirt,a cd cover,a dvd cover,a book cover,a video game cover,a photo of shoes",
        help="phrases to filter out unwanted samples using CLIP similiarity (phrases separated by commas). Set to empty string '' to bypass filtering."
    )
    parser.add_argument(
        "--filter_phrase_thresholds",
        type=str,
        default="17.5,17.5,16.0,18.0,18.0,18.0,18.0,18.0,17.0",
        help="Scalar thresholds for each filter phrases, separated by commas.",
    )
    parser.add_argument("--min_orig_image_size", default=200, type=int, help="minimum height/width to include image")
    parser.add_argument("--bev_detect_thresh", type=float, default=0.08, help="threshold for detecting person with bev heatmap")
    parser.add_argument("--bev_low_x_thresh", type=float, default=0.0, help="low threshold for depth filtering (remove poses below threshold)")
    parser.add_argument("--bev_high_x_thresh", type=float, default=120.0, help="high threshold for depth filtering (remove poses above threshold)")
    parser.add_argument("--bev_low_y_thresh", type=float, default=0.0, help="low threshold for depth filtering (remove poses below threshold)")
    parser.add_argument("--bev_high_y_thresh", type=float, default=120.0, help="high threshold for depth filtering (remove poses above threshold)")
    parser.add_argument("--bev_low_z_thresh", type=float, default=4.0, help="low threshold for depth filtering (remove poses below threshold)")
    parser.add_argument("--bev_high_z_thresh", type=float, default=50.0, help="high threshold for depth filtering (remove poses above threshold)")
    args = parser.parse_args()

    # folder to save viz results
    out_path = os.path.join(args.out_path, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    os.makedirs(out_path)
    os.makedirs(os.path.join(out_path, 'samples'))
    os.makedirs(os.path.join(out_path, 'samples_removed'))
    os.makedirs(os.path.join(out_path, 'samples_removed_json'))

    laion_data = Laion(
                    args.laion_tar_files,
                    preprocess_type="bev_blip2",
                    min_orig_image_size=args.min_orig_image_size,
                )
    data_loader = laion_data.wds_loader(
        batch_size=args.batch_size,
        resampled=True,
        shuffle=2000,
    )
    data_iterator = iter(data_loader)
    for _ in range(args.burnin_batches):
        _ = next(data_iterator)

    bev_model = import_bev(torch.device(0))
    # set threshold for person detection with bev heatmap
    bev_model.model.centermap_parser.conf_thresh = args.bev_detect_thresh
    # set the near/far thresholds for filtering out pose predictions (only keep predictions in middle range)
    bev_model.low_x_thresh = args.bev_low_x_thresh
    bev_model.high_x_thresh = args.bev_high_x_thresh
    bev_model.low_y_thresh = args.bev_low_y_thresh
    bev_model.high_y_thresh = args.bev_high_y_thresh
    bev_model.low_z_thresh = args.bev_low_z_thresh
    bev_model.high_z_thresh = args.bev_high_z_thresh

    clip_model = get_clip_model(args.clip_type, (args.clip_ckpt if hasattr(args, 'clip_ckpt') else None))

    if not args.filter_phrases == "":
        filter_phrases = args.filter_phrases.split(",")
        filter_phrase_thresholds = [float(thresh) for thresh in args.filter_phrase_thresholds.split(",")]
        assert len(filter_phrases) == len(filter_phrase_thresholds)

    for i, batch in enumerate(data_iterator):
        print(f'Scanning batch {i+1}')
        if i >= args.max_batches:
            break

        if args.clip_type == 'orig':
            image_clip = batch['image_clip_vit']
        else:
            image_clip = batch['image_open_clip']
        image_bev = batch['image_bev']
        padding_bev = batch['padding_bev']
        pass_filter_json = batch['pass_filter']
        key = batch["key"]

        # bev result for batch of images
        batch_outputs = bev_model(image_bev.cuda(), padding_bev)

        for j in range(len(batch_outputs)):
            if batch_outputs[j] is not None:
                pass_filter = pass_filter_json[j]
                if not args.filter_phrases == "" and pass_filter:
                    pass_filter, filter_str = apply_clip_filter(
                                    image_clip=image_clip[j:(j+1)],
                                    filter_phrases=filter_phrases,
                                    filter_phrase_thresholds=filter_phrase_thresholds,
                                    clip_type=args.clip_type,
                                    clip_model=clip_model,
                                )
                elif not pass_filter:
                    filter_str = "json"

                sample_out_path = (
                    os.path.join(out_path, "samples") if pass_filter
                    else (
                        os.path.join(out_path, "samples_removed_json") if filter_str == 'json'
                        else os.path.join(out_path, "samples_removed")
                    )
                )
                viz_image(
                    image_np=batch_outputs[j]['rendered_image'][:, :, ::-1],
                    caption=f"filter: {filter_str}",
                    out_path=sample_out_path,
                    id=key[j]
                )
