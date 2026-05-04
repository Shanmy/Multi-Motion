import os
import json
import tarfile

import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

import webdataset as wds

import clip

from bev.post_parser import body_mesh_projection2image


def detect_person_detectron2(detectron2_model, image_cv2):
    outputs = detectron2_model(image_cv2.cpu().numpy()[0])
    return (0 in outputs["instances"].pred_classes.cpu().numpy())  # NOTE: class 0 is person in COCO annotations


def caption_image_blip2(args, blip2_model, blip2_processor, clip_model_vit, clip_model_res,
                        image_blip2, image_clip_vit, image_clip_res, pose_query=""):
    if pose_query == "":
        # generate caption from only image features
        inputs = blip2_processor(image_blip2, return_tensors="pt").to(image_blip2.device, torch.float16)
    else:
        # generate caption from image with query text
        inputs = blip2_processor(image_blip2, pose_query, return_tensors="pt").to(image_blip2.device, torch.float16)     

    max_logits = torch.tensor([-1e3] * args.blip2_round_1_size)
    max_logits = max_logits.unsqueeze(0).repeat(image_blip2.shape[0], 1)
    blip2_texts = [[''] * args.blip2_round_1_size for _ in range(image_blip2.shape[0])]

    for _ in range(args.blip2_reps):

        # random generation of decoding params
        # https://github.com/andreaskoepf/CLIP-Image-Captioning/blob/blip_test/blip_coco_val_sample_sweep_bayes_02.yaml
        min_length = torch.randint(low=args.blip2_min_length, high=args.blip2_max_length, size=[1]).item()
        top_k = torch.randint(low=100, high=10000, size=[1]).item()
        top_p = (0.01 + 0.99 * torch.rand(size=[1])).item()

        generated_ids = blip2_model.generate(
            **inputs,
            do_sample=True,
            max_length=args.blip2_max_length,
            min_length=min_length,
            top_k=top_k,
            top_p=top_p,
        )
        blip2_text_batch = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)
        blip2_text_batch = [x.strip() for x in blip2_text_batch] # list of length batch_size

        clip_text = clip.tokenize(blip2_text_batch, truncate=True).to(image_clip_vit.device) # batch_size x 77
        logit_vit, _ = clip_model_vit(image_clip_vit, clip_text) 
        logit_vit = torch.diagonal(logit_vit)
        # optional small penalty for each token to encourage more concise caption
        logit_vit -= args.clip_len_penalty * clip_text.nonzero().shape[0]

        # generate many captions and select top n as ranked by clip vit
        for img_idx in range(image_blip2.shape[0]):
            if logit_vit[img_idx] > max_logits[img_idx].min():
                min_logit_ind = torch.where(max_logits[img_idx] == max_logits[img_idx].min())[0][0].item()
                blip2_texts[img_idx][min_logit_ind] = blip2_text_batch[img_idx]
                max_logits[img_idx][min_logit_ind] = logit_vit[img_idx].squeeze().cpu()

    blip2_text = ['' for i in range(image_blip2.shape[0])]
    candidate_texts = []
    max_logit = [-1e3 for _ in range(image_blip2.shape[0])]

    # use clip resnet to select best caption out of top n
    for cap_idx in range(args.blip2_round_1_size):
        candidate_texts = [blip2_texts[img_idx][cap_idx] for img_idx in range(image_blip2.shape[0])]
        clip_text = clip.tokenize(candidate_texts, truncate=True).to(image_clip_res.device)
        logit_res, _ = clip_model_res(image_clip_res, clip_text)
        logit_res = torch.diagonal(logit_res)
        logit_res -= args.clip_len_penalty * clip_text.nonzero().shape[0]

        for img_idx in range(image_blip2.shape[0]):
            if logit_res[img_idx] > max_logit[img_idx]:
                max_logit[img_idx] = logit_res[img_idx]
                blip2_text[img_idx] = candidate_texts[img_idx]

    blip2_text = np.array(blip2_text)

    return blip2_text


def caption_image_instruct_blip(
    instruct_blip_model,
    instruct_blip_processor,
    instruction_prompt,
    images,
    batch_size,
    deterministic_caption=True,
    max_length=77,
    min_length=0,  # used for filtering out samples with short captions, not for limiting caption length
):

    def get_output_length(output):
        # token 1 marks the end of the sequence
        last_token_idx = (output == 1).nonzero()
        if len(last_token_idx) > 0:
            output_length = int(last_token_idx[0][0].cpu()) + 1
        else:
            output_length = max_length
        return output_length

    generated_texts = []
    pass_filter_instruct_blip = []
    batch = 0

    while len(generated_texts) < images.shape[0]:

        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, images.shape[0])
        batch += 1

        images_batch = images[start_idx:end_idx]
        instruction_prompt_rep = images_batch.shape[0] * [instruction_prompt]
        inputs = instruct_blip_processor(images=images_batch, text=instruction_prompt_rep, return_tensors="pt")
        inputs = inputs.to(images.device, dtype=torch.bfloat16)
        if deterministic_caption:
            outputs = instruct_blip_model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=5,
                    max_length=max_length,
                    min_length=1,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                )
        else:
            outputs = instruct_blip_model.generate(
                    **inputs,
                    do_sample=True,
                    num_beams=5,
                    max_length=max_length,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                )

        # overwrite -1 output tokens with 0, otherwise decoder will fail with error: 
        # "OverflowError: out of range integral type conversion attempted"
        outputs[outputs == -1] = 0

        pass_filter_instruct_blip_batch = [
            1 if get_output_length(output) >= min_length else 0 for output in outputs
        ]
        pass_filter_instruct_blip.extend(pass_filter_instruct_blip_batch)

        generated_text = instruct_blip_processor.batch_decode(outputs, skip_special_tokens=True)
        generated_text = [x.strip() for x in generated_text]
        generated_texts.extend(generated_text)

    generated_texts = np.array(generated_texts)
    pass_filter_instruct_blip = np.array(pass_filter_instruct_blip)

    return generated_texts, pass_filter_instruct_blip


def caption_video_from_frame_captions(clip_model_res, frame_captions, frame_dataset, 
                                      vid_cap_percentile, device, batch_size=25):
    # TODO: replace average CLIP score with EMScore captioning quality metric?
    cap_score_mat = torch.zeros([len(frame_captions), len(frame_dataset)])
    frame_loader = DataLoader(frame_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    for idx, caption in enumerate(frame_captions):
        print('Evaluating caption: ', caption)
        frame_iterator = iter(frame_loader)
        for batch_idx, batch in enumerate(frame_iterator):
            clip_text = clip.tokenize([caption], truncate=True).to(device)
            _, _, _, image_clip_res, _ = batch
            image_clip_res = image_clip_res.to(device)
            logit, _ = clip_model_res(image_clip_res, clip_text)

            batch_start = batch_idx * batch_size
            batch_end = min(len(frame_dataset), (batch_idx + 1) * batch_size)
            cap_score_mat[idx, batch_start:batch_end] = logit.detach().squeeze().cpu()

    cap_scores = cap_score_mat.mean(dim=1)
    caption_index = int(vid_cap_percentile * len(cap_scores)) - 1
    video_caption = frame_captions[torch.sort(cap_scores).indices[caption_index].item()]
    print('Final Video Caption: ', video_caption)
    return video_caption


def clip_filter(
    image_clip,
    clip_model,
    filter_phrases,
    filter_phrase_thresholds,
    clip_tokenizer=None,
    clip_type='orig',
):
    if clip_tokenizer is None:
        clip_tokenizer = clip.tokenize
    clip_tokens = clip_tokenizer(filter_phrases).cuda() # batch_size x 77

    if clip_type == 'orig':
        logit, _ = clip_model(image_clip, clip_tokens)
    elif clip_type == 'open_clip':
        out_dict = clip_model(image_clip, clip_tokens)
        logit = out_dict['logit_scale'] * out_dict['image_features'] @ out_dict['text_features'].T
    else:
        raise ValueError("Invalid 'clip_type' (should be 'orig' or 'open_clip')")

    pass_filter = True

    if pass_filter:
        closest_filter_phrase = ""
        smallest_logit_dist = None
        for i in range(len(filter_phrases)):
            if smallest_logit_dist is None or filter_phrase_thresholds[i] - logit[0][i] < smallest_logit_dist:
                smallest_logit_dist = filter_phrase_thresholds[i] - logit[0][i]
                logit_score_str = str(logit[0][i].item()) + "/" + str(filter_phrase_thresholds[i])
                closest_filter_phrase = filter_phrases[i] + " " + logit_score_str
            if logit[0][i] > filter_phrase_thresholds[i]:
                pass_filter = False

    return pass_filter, closest_filter_phrase


def learned_clip_filter(
    image_open_clip,
    open_clip_model,
    linear_model,
    threshold,
):

    out_dict = open_clip_model(image_open_clip)
    clip_embedding = out_dict['image_features'].cpu().numpy()
    lm_prob = linear_model.predict_proba(clip_embedding)
    pass_filter = lm_prob[0, 1] >= threshold

    return pass_filter


def remove_out_of_frame_poses(outputs, padding_bev, smpl_parser, completion_requirement=0.8):
    # order of padding_bev: [top_pad, bottom_pad, left_pad, right_pad, height, width]

    # get estimated meshes
    verts, joints, face = smpl_parser(outputs['smpl_betas'], outputs['smpl_thetas'])
    outputs.update({'verts': verts, 'joints': joints, 'smpl_face': face})
    outputs.update(
        body_mesh_projection2image(
            outputs['joints'],
            outputs['cam_trans'],
            vertices=outputs['verts'],
            input2org_offsets=padding_bev,
            denormalize_trans=False,
        )
    )

    smpl_betas_out = []
    smpl_thetas_out = []
    cam_trans_out = []

    for i, vertex_sample in enumerate(outputs['verts_camed_org'].cpu().numpy()):

        # remove pose predictions where a large proportion of mesh falls outside of image
        vertical_filter = (vertex_sample[:, 0] < 0.0) | (vertex_sample[:, 0] > padding_bev[5])
        horizontal_filter = (vertex_sample[:, 1] < 0.0) | (vertex_sample[:, 1] > padding_bev[4])
        boundary_filter = horizontal_filter | vertical_filter
        if 1 - (boundary_filter.sum() / vertex_sample.shape[0]) >= completion_requirement:
            smpl_betas_out.append(outputs['smpl_betas'][i])
            smpl_thetas_out.append(outputs['smpl_thetas'][i])
            cam_trans_out.append(outputs['cam_trans'][i])

    if smpl_betas_out == []:
        outputs = None
    else:
        # repack output dictionary with only poses that are (mostly) inside the image frame
        outputs['smpl_betas'] = torch.stack(smpl_betas_out)
        outputs['smpl_thetas'] = torch.stack(smpl_thetas_out)
        outputs['cam_trans'] = torch.stack(cam_trans_out)
    return outputs


def save_samples_to_tarfile(sample_list, tarfile_save_path):
    with wds.TarWriter(tarfile_save_path) as sink:
        for i, sample in enumerate(sample_list):
            # add info about tarfile and local sample id
            sample["shard_info.json"] = dict(
                local_id=i,
                shard_id=os.path.basename(tarfile_save_path),
            )
            # save sample in tarfile
            sink.write(sample)


def sync_tensor_across_workers(tensor):
    tensor_gather = [
        torch.zeros(tensor.shape, device=tensor.device, dtype=tensor.dtype)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(tensor_gather, tensor)
    return torch.stack(tensor_gather)
