import os.path as osp
import glob
import json
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, Pad
)


class CustomDataset(Dataset):
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        # get image list by scanning for suitable files in folder
        self.image_list = glob.glob(osp.join(dataset_path, '**/*.jpg'), recursive=True)
        self.image_list.extend(glob.glob(osp.join(dataset_path, '**/*.png'), recursive=True))
        self.image_list.extend(glob.glob(osp.join(dataset_path, '**/*.jpeg'), recursive=True))
        self.image_list.extend(glob.glob(osp.join(dataset_path, '**/*.JPEG'), recursive=True))
        self.list_size = len(self.image_list)

        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        # https://www.grepper.com/answers/353879/pytorch+pad+to+square
        class SquarePad:
            def __call__(self, image):
                max_wh = max(image.size)
                p_left, p_top = [(max_wh - s) // 2 for s in image.size]
                p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
                padding = (p_left, p_top, p_right, p_bottom)
                return Pad(padding)(image)

        # resize to 224x224, center crop, normalize
        self.clip_transforms_vit = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        # resize to 448x448, center crop, normalize
        self.clip_transforms_res = Compose([
            Resize(448, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(448),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        # pad to square, resize to 512x512, normalize to pixel range [-1, 1]
        self.bev_transforms = Compose([
            _convert_image_to_rgb,
            SquarePad(),
            Resize(512, interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.blip2_transform = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            ToTensor()
        ])

    def _get_pad(self, image):
        w, h = image.size
        side_length = max(w, h)
        top, left = int((side_length - h) // 2), int((side_length - w) // 2)
        bottom, right = int(top+h), int(left+w)
        image_pad_info = torch.Tensor([top, bottom, left, right, h, w])
        return image_pad_info

    def __getitem__(self, index):

        image_path = self.image_list[index].rstrip()

        image_name = image_path.replace(self.dataset_path, '').lstrip('/')
        image_pil = Image.open(image_path)

        image_blip2 = 255 * self.blip2_transform(image_pil)
        if image_blip2.shape[0] == 1:
            image_blip2 = image_blip2.repeat(3, 1, 1)
        image_bev = self.bev_transforms(image_pil)
        image_clip_vit = self.clip_transforms_vit(image_pil)
        image_clip_res = self.clip_transforms_res(image_pil)
        padding = self._get_pad(image_pil)

        return image_bev, image_blip2, image_clip_vit, image_clip_res, padding, image_path

    def __len__(self):
        return self.list_size
