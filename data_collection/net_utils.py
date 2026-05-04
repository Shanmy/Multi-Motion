import torch

import os.path as osp
import sys
import os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.join(_here, 'open_clip/src/'))
sys.path.insert(0, _os.path.join(_here, 'ROMP/simple_romp/'))

import bev
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
import clip
from open_clip.factory import create_model_and_transforms
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


# default net folders on starfire and local server
LOCAL_NET_PATH = '/nfs/USRCSEA/IVA/Models/PoseCaptionData/'
STARFIRE_NET_PATH = '/home/notebook/data/group/pose_caption_dataset/'
# default net locations in net folder
DEFAULT_BEV_PATH = 'ROMP_and_BEV/'
DEFAULT_BLIP2_PATH = ''
DEFAULT_INSTRUCT_BLIP_PATH = ''
DEFAULT_CLIP_PATH = 'clip/'
DEFAULT_OPEN_CLIP_PATH = 'open_clip/vit_h_14_open_clip_pytorch_model.bin'
DEFAULT_DETECTRON2_PATH = 'detectron2/model_final_f10217.pkl'


def import_bev(device, bev_path=None, render=True, use_starfire=False):

    if bev_path is None:
        net_path = STARFIRE_NET_PATH if use_starfire else LOCAL_NET_PATH
        bev_path = osp.join(net_path, DEFAULT_BEV_PATH)

    bev_settings = bev.main.default_settings
    bev_settings.crowd = False  # https://github.com/Arthur151/ROMP/issues/293
    bev_settings.model_path = osp.join(bev_path, 'BEV.pth')
    bev_settings.smpl_path = osp.join(bev_path, 'smpla_packed_info.pth')
    bev_settings.smil_path = osp.join(bev_path, 'smil_packed_info.pth')
    bev_settings.render_mesh = render
    bev_model = bev.BEV(bev_settings, device)
    return bev_model

def import_blip2(device, blip2_path=None, use_starfire=False):

    if blip2_path is None:
        net_path = STARFIRE_NET_PATH if use_starfire else LOCAL_NET_PATH
        blip2_path = osp.join(net_path, DEFAULT_BLIP2_PATH)

    blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        cache_dir=blip2_path,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    blip2_model.to(device)
    blip2_model.eval()
    blip2_processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        cache_dir=blip2_path,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    return blip2_model, blip2_processor

def import_instruct_blip(device, instruct_blip_path=None, use_starfire=False):

    if instruct_blip_path is None:
        net_path = STARFIRE_NET_PATH if use_starfire else LOCAL_NET_PATH
        instruct_blip_path = osp.join(net_path, DEFAULT_INSTRUCT_BLIP_PATH)

    # load net and data processor
    instruct_blip_model = InstructBlipForConditionalGeneration.from_pretrained(
                "Salesforce/instructblip-vicuna-13b",
                cache_dir=instruct_blip_path,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
    instruct_blip_model.to(device, dtype=torch.bfloat16)
    instruct_blip_model.eval()

    # edit embedding layers to overwite input_ids -1 and replace with 0.
    # this prevents huggingface transformer error for LLAMA using batch_size > 1
    class EmbeddingEdit(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embedding = embedding
        def forward(self, input_ids):
            input_ids[input_ids == -1] = 0
            return self.embedding(input_ids)
    embed_tokens_edit = EmbeddingEdit(instruct_blip_model.language_model.model.embed_tokens)
    instruct_blip_model.language_model.model.embed_tokens = embed_tokens_edit

    instruct_blip_processor = InstructBlipProcessor.from_pretrained(
                "Salesforce/instructblip-vicuna-13b",
                cache_dir=instruct_blip_path,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
    return instruct_blip_model, instruct_blip_processor

def import_clip(device, clip_path=None, clip_type="ViT-L/14", use_starfire=False):

    if clip_path is None:
        net_path = STARFIRE_NET_PATH if use_starfire else LOCAL_NET_PATH
        clip_path = osp.join(net_path, DEFAULT_CLIP_PATH)

    clip_model, _ = clip.load(
        clip_type,
        device=device,
        download_root=clip_path,
    )
    clip_model.eval()
    return clip_model

def import_open_clip(device, open_clip_path=None, open_clip_type='ViT-H-14', use_starfire=False):

    if open_clip_path is None:
        net_path = STARFIRE_NET_PATH if use_starfire else LOCAL_NET_PATH
        open_clip_path = osp.join(net_path, DEFAULT_OPEN_CLIP_PATH)

    open_clip_model, _, _ = create_model_and_transforms(
        open_clip_type,
        open_clip_path,
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

    return open_clip_model

def import_detectron2(device, detectron2_path=None, detectron2_thresh=0.5, use_starfire=False,
                      detectron2_config="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):

    if detectron2_path is None:
        net_path = STARFIRE_NET_PATH if use_starfire else LOCAL_NET_PATH
        detectron2_path = osp.join(net_path, DEFAULT_DETECTRON2_PATH)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(detectron2_config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detectron2_thresh  # set threshold for this model
    cfg.MODEL.WEIGHTS = detectron2_path
    predictor = DefaultPredictor(cfg)
    return predictor

def import_pose_and_caption_nets(
    device,
    bev_path=None,
    blip2_path=None,
    instruct_blip_path=None,
    clip_path=None,
    open_clip_path=None,
    detectron2_path=None,
    render=True,
    use_instruct_blip=False,
    use_starfire=False,
):

    bev_model = import_bev(device, bev_path, render=render, use_starfire=use_starfire)
    blip2_model, blip2_processor = import_blip2(device, blip2_path, use_starfire=use_starfire)
    if use_instruct_blip:
        instruct_blip_model, instruct_blip_processor = \
            import_instruct_blip(device, instruct_blip_path, use_starfire=use_starfire)
    else:
        instruct_blip_model, instruct_blip_processor = None, None
    clip_model_vit = import_clip(device, clip_path, clip_type="ViT-L/14", use_starfire=use_starfire)
    clip_model_res = import_clip(device, clip_path, clip_type="RN50x64", use_starfire=use_starfire)
    open_clip_model = import_open_clip(device, open_clip_path, use_starfire=use_starfire)
    detectron2_model = import_detectron2(device, detectron2_path, use_starfire=use_starfire)

    return (
        bev_model,
        blip2_model,
        blip2_processor,
        instruct_blip_model,
        instruct_blip_processor,
        clip_model_vit,
        clip_model_res,
        open_clip_model,
        detectron2_model,
    )
