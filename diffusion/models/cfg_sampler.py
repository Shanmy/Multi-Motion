import torch
import torch.nn as nn
from copy import deepcopy
import random


# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, guidance_param=2.5):
        super().__init__()
        self.model = model  # model is the actual model to run
        if hasattr(self.model, 'encode_text'):
            self.encode_text = self.model.encode_text
        if hasattr(self.model, 'update_cond_dict'):
            self.update_cond_dict = self.model.update_cond_dict
        if hasattr(self.model, 'denoised_fn'):
            self.denoised_fn = self.model.denoised_fn

        self.guidance_param = guidance_param
        assert self.guidance_param > 1.0

    def forward(self, x, timesteps, **cond_dict):

        uncond_dict = deepcopy(cond_dict)
        uncond_dict['text'] = [''] * x.shape[0]
        if 'text_encoding_uncond' in cond_dict.keys():
            uncond_dict['text_encoding'] = cond_dict['text_encoding_uncond']
        out = self.model(x, timesteps, **cond_dict)
        out_uncond = self.model(x, timesteps, **uncond_dict)
        scale = self.guidance_param * torch.ones(x.shape).cuda()

        return out_uncond + scale * (out - out_uncond)


def apply_text_cond_mask(text, cond_mask_prob):
    for i in range(len(text)):
        if (random.random() < cond_mask_prob):
            text[i] = ''
    return text
