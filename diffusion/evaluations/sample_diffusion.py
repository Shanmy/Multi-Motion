import numpy as np

import torch
import torch.distributed as dist

from pose_datasets.data_utils import unnormalize, set_dummy_poses_to_zero
from evaluations.denoised_fn import get_denoised_fn


def sample_diffusion(
    model,
    diffusion,
    num_samples,
    local_rank,
    dataloader,  # currently only support samples using conds from dataloader. to use new prompts, make dummy dataloader.
    mean_path=None,
    std_path=None,
    num_sample_frames=None,
    cond_model_dict=None,
):

    sample_fn = diffusion.p_sample_loop if not diffusion.use_ddim else diffusion.ddim_sample_loop

    count = 0
    samples = []
    # save sampling conditions that are needed for viz or eval
    cond_out = None

    while count < num_samples:

        batch, cond = next(dataloader)
        if 'text' in cond.keys():
            # embed the text prompts one time before sampling starts. saves time.
            cond['text_encoding'] = model.encode_text(cond['text'])
            # for cfg etc.
            dummy_text = [""] * len(cond['text'])
            cond['text_encoding_uncond'] = model.encode_text(dummy_text)
        if num_sample_frames is not None:
            # overwrite lengths argument with different number of frames
            cond['lengths'] = torch.tensor([num_sample_frames] * batch.shape[0]).long()
            sample_shape = [batch.shape[0], num_sample_frames, batch.shape[-2], batch.shape[-1]]
        else:
            sample_shape = list(batch.shape)
        # put on gpu
        cond = {
            k: v.to(local_rank) if torch.is_tensor(v) else v
            for k, v in cond.items()
        }
        if hasattr(model, 'update_cond_dict'):
            # sampling conditioned on data
            cond = model.update_cond_dict(batch.to(local_rank), cond)

        model_denoised_fn = model.denoised_fn if hasattr(model, 'denoised_fn') else None
        denoised_fn = get_denoised_fn(cond_model_dict=cond_model_dict, model_denoised_fn=model_denoised_fn)
        # get sample
        sample = sample_fn(
                model,
                sample_shape,
                clip_denoised=False,
                denoised_fn=denoised_fn,
                model_kwargs=cond,
            )
        sample = sample.cpu()

        if mean_path is not None:
            sample = unnormalize(sample, mean_path, std_path)
        sample = set_dummy_poses_to_zero(
                sample,
                lengths=(cond['lengths'] if 'lengths' in cond.keys() else None),
                num_poses=(cond['num_poses'] if 'num_poses' in cond.keys() else None),
            )

        samples.append(sample)
        count += sample.shape[0]

        if cond_out is None:
            # init as empty list for each key in the dataloader conds after first batch
            cond_out = {k: [] for k in cond.keys()}
        if 'text' in cond.keys():
            cond_out['text'] += cond['text']
        if 'lengths' in cond.keys():
            cond_out['lengths'] += cond['lengths']
        if 'num_poses' in cond.keys():
            cond_out['num_poses'] += cond['num_poses']

    samples = torch.cat(samples)

    return samples, cond_out
