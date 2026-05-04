import os
import tarfile
import json
import braceexpand

import numpy as np
import cv2

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, Pad
)

import webdataset as wds


class Laion:
    def __init__(
        self,
        urls="/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data4/{00000..32449}.tar",
        preprocess_type="standard",  # "standard" or "bev_blip2"
        image_size=256,
        min_orig_image_size=200,
    ):

        self.urls = list(braceexpand.braceexpand(urls))
        self.preprocess_type = preprocess_type

        def convert_to_rgb(image):
            return image.convert("RGB")
        self.convert_to_rgb = convert_to_rgb

        if self.preprocess_type == "standard":
            self._init(image_size=image_size)
        elif self.preprocess_type == "bev_blip2":
            self.min_orig_image_size = min_orig_image_size
            self._init_bev_blip2()
        else:
            raise ValueError("Invalid 'preprocess_type'.")

    def _init(self, image_size=256):

        self.transforms = Compose([
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            self.convert_to_rgb,
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        def preprocess(sample):
            image, json_info, _ = sample
            image = self.transforms(image)
            try:
                text = json_info["caption"]
            except:
                text = ""
            return image, text
        
        self.preprocess = preprocess

    def _init_bev_blip2(self):
        # initialize image transforms for bev/blip2 pose and caption collection

        # https://www.grepper.com/answers/353879/pytorch+pad+to+square
        class SquarePad:
            def __call__(self, image):
                max_wh = max(image.size)
                p_left, p_top = [(max_wh - s) // 2 for s in image.size]
                p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
                padding = (p_left, p_top, p_right, p_bottom)
                return Pad(padding)(image)

        self.bev_transforms = Compose([
            self.convert_to_rgb,
            SquarePad(),
            Resize(512, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(512),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.blip2_transform = Compose([
            self.convert_to_rgb,
            SquarePad(),
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            ToTensor()
        ])

        self.clip_transforms_vit = Compose([
            self.convert_to_rgb,
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.clip_transforms_res = Compose([
            self.convert_to_rgb,
            Resize(448, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(448),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.detectron2_transforms = Compose([
            self.convert_to_rgb,
            SquarePad(),
            Resize(256, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(256),
            ToTensor(),
        ])

        self.open_clip_transforms = Compose([
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                self.convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        def _get_pad(image):
            w, h = image.size
            side_length = max(w, h)
            top, left = int((side_length - h) // 2), int((side_length - w) // 2)
            bottom, right = int(top+h), int(left+w)
            image_pad_info = torch.Tensor([top, bottom, left, right, h, w])
            return image_pad_info

        def json_filter(json_info):
            pass_filter = True
            if not json_info["NSFW"] == "UNLIKELY":
                pass_filter = False
            if json_info["original_height"] < self.min_orig_image_size:
                pass_filter = False
            if json_info["original_width"] < self.min_orig_image_size:
                pass_filter = False
            return pass_filter

        def preprocess(sample):
            image, json_info, url = sample

            key = json_info['key']
            pass_filter = json_filter(json_info)

            image_pil = image
            image_blip2 = 255 * self.blip2_transform(image_pil)
            if image_blip2.shape[0] == 1:
                image_blip2 = image_blip2.repeat(3, 1, 1)
            image_bev = self.bev_transforms(image_pil)
            image_clip_vit = self.clip_transforms_vit(image_pil)
            image_clip_res = self.clip_transforms_res(image_pil)
            image_open_clip = self.open_clip_transforms(image_pil)
            image_detectron2 = 255 * self.detectron2_transforms(image_pil)[[2, 1, 0]].permute(1, 2, 0)

            try:
                text = json_info["caption"]
            except:
                text = ""

            # bev padding might be needed for viz
            padding_bev = _get_pad(image_pil)

            # Check if all variables are not None
            variables_to_validate = [
                image_bev,
                image_blip2,
                image_clip_res,
                image_clip_vit,
                image_open_clip,
                image_detectron2,
                text,
                key,
                url,
                pass_filter,
                padding_bev,
            ]
            valid_flag = all(var is not None for var in variables_to_validate)

            # manually overwrite None data with dummy data of the expected type and shape
            if image_bev is None:
                image_bev = torch.randn([3, 512, 512])
            if image_blip2 is None:
                image_blip2 = torch.randn([3, 224, 224])
            if image_clip_res is None:
                image_clip_res = torch.randn([3, 448, 448])
            if image_clip_vit is None:
                image_clip_vit = torch.randn([3, 224, 224])
            if image_open_clip is None:
                image_open_clip = torch.randn([3, 224, 224])
            if image_detectron2 is None:
                image_detectron2 = torch.randn([256, 256, 3])
            if text is None:
                text = ""
            if key is None:
                key = "-1"
            if url is None:
                url = ""
            if pass_filter is None:
                pass_filter = False
            if padding_bev is None:
                padding_bev = torch.randn([6])

            return {
                'image_bev': image_bev,
                'image_blip2': image_blip2,
                'image_clip_res': image_clip_res,
                'image_clip_vit': image_clip_vit,
                'image_open_clip': image_open_clip,
                'image_detectron2': image_detectron2,
                'text': text,
                'key': key,
                'url': url,
                'pass_filter': pass_filter,
                'padding_bev': padding_bev,
                'mask': valid_flag,
            }

        self.preprocess = preprocess

    def wds_loader(
        self,
        batch_size,
        num_workers=1,
        resampled=False,
        shuffle=2000,
    ):
        dataset = wds.WebDataset(
            self.urls,
            resampled=resampled,
            handler=wds.ignore_and_continue,
            nodesplitter=nodesplitter,
        )
        if shuffle > 0:
            dataset = dataset.shuffle(shuffle)
        dataset = dataset.decode("pil", handler=wds.ignore_and_continue)
        dataset = dataset.to_tuple("jpg", "json", "__url__").map(self.preprocess)
        if not self.preprocess_type == "bev_blip2":
            dataset = dataset.batched(batch_size, partial=False)
            batch_size = None
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return data_loader


def nodesplitter(src, group=None):
    if dist.is_initialized():
        if group is None:
            group = dist.group.WORLD
        rank = dist.get_rank(group=group)
        size = dist.get_world_size(group=group)
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src


def load_img_tar(key, laion_path="/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data4/"):
    # NOTE: each laion file has unique ID in numerical order of data file.
    # also, each tarfile has 10K samples.
    # therefore the first 5 digits of each laion file ID gives the tarfile ID.

    tar_file_path = os.path.join(laion_path, f"{int(key[:5]):05d}.tar")
    image_filename = f"{int(key):09d}.jpg"

    with tarfile.open(tar_file_path, 'r') as tar:
        # Extract the image file from the tar archive
        image_data = tar.extractfile(image_filename)

        if image_data is not None:
            # Read the image using OpenCV
            image_np = np.frombuffer(image_data.read(), np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

    return image


def load_json_tar(key, laion_path="/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data4/"):
    tar_file_path = f"{laion_path}{int(key[:5]):05d}.tar"
    json_filename = f"{int(key):09d}.json"

    with tarfile.open(tar_file_path, 'r') as tar:
        with tar.extractfile(json_filename) as f:
            json_data = json.loads(f.read())

    return json_data


# for reference, can be added to pipeline using .select()
def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image
