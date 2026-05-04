import os
import argparse
from datetime import datetime
import time

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

import torch
import torch.distributed as dist

from data_laion import Laion
from net_utils import import_pose_and_caption_nets
from processing_utils import (
    detect_person_detectron2,
    caption_image_blip2,
    caption_image_instruct_blip,
    remove_out_of_frame_poses,
    clip_filter,
    learned_clip_filter,
    save_samples_to_tarfile,
    sync_tensor_across_workers,
)

from viz.viz_utils import viz_image
from viz.viz_bev import viz_render_bev


def save_code_files(out_path):

    # save a single file
    def save_file(file_in_name, file_out_name):
        file_in = open(file_in_name, 'r')
        file_out = open(file_out_name, 'w')
        for line in file_in:
            file_out.write(line)

    # folders containing files needed to reproduce data collection
    save_folder_list = [
        './',
        'ROMP/simple_romp/romp',
        'ROMP/simple_romp/bev',
    ]
    for folder in save_folder_list:
        if not folder == './':
            os.makedirs(os.path.join(out_path, folder))
        folder_files = os.listdir(folder)
        for folder_file in folder_files:
            if folder_file.endswith('.py') or folder_file.endswith('.sh'):
                in_file = os.path.join(folder, folder_file)
                out_file = os.path.join(out_path, folder, folder_file)
                save_file(in_file, out_file)


def create_tarfiles(
    sample_list,
    current_tarfile_id,
    out_path,
    samples_per_tarfile,
    world_rank,
    device,
    world_size,
    tarname_num_digits=5,
):
    # sync file name across gpu and save samples for workers that have a large enough sample list
    for worker_rank in range(world_size):
        # coordinate new tarfile id across workers
        current_tarfile_id_gpu = torch.tensor([current_tarfile_id]).long().to(device)
        current_tarfile_id_gather = sync_tensor_across_workers(current_tarfile_id_gpu)
        current_tarfile_id = int(current_tarfile_id_gather.max().cpu().numpy())
        save_path = os.path.join(out_path, str(current_tarfile_id).zfill(tarname_num_digits) + '.tar')

        # for each worker, save sample list if full
        if len(sample_list) >= samples_per_tarfile and world_rank == worker_rank:
            print(f'Device {world_rank} Saving Tarfile: {save_path}')
            sample_list_save = sample_list[:samples_per_tarfile]
            sample_list = sample_list[samples_per_tarfile:]
            assert len(sample_list_save) == samples_per_tarfile
            save_samples_to_tarfile(sample_list_save, save_path)
            current_tarfile_id += 1

        dist.barrier()
    dist.barrier()

    return sample_list, current_tarfile_id


def collect_data(args, world_rank, device, world_size):

    # output directory
    if world_rank == 0:
        os.makedirs(args.out_path)
        if args.render_selected:
            # samples selected for dataset inclusion
            os.makedirs(os.path.join(args.out_path, 'selected'))
        if args.render_orig:
            # original bev results for selected samples
            os.makedirs(os.path.join(args.out_path, 'good'))
            # samples removed for various reasons
            os.makedirs(os.path.join(args.out_path, 'json'))
            os.makedirs(os.path.join(args.out_path, 'detectron'))
            os.makedirs(os.path.join(args.out_path, 'mesh'))
            os.makedirs(os.path.join(args.out_path, 'handcraft'))
            os.makedirs(os.path.join(args.out_path, 'learned'))
            os.makedirs(os.path.join(args.out_path, 'instruct'))

    # initialize models
    (
        bev_model,
        blip2_model,
        blip2_processor,
        instruct_blip_model,
        instruct_blip_processor,
        clip_model_vit,
        clip_model_res,
        open_clip_model,
        detectron2_model,
    ) = import_pose_and_caption_nets(
        device,
        use_instruct_blip=args.use_instruct_blip,
        render=args.render_orig
    )

    # learn linear model for clip filtering
    df = pd.read_csv(args.filter_dataset_path)
    clip_embeddings = np.load(args.filter_embeddings_path)
    linear_model = LogisticRegression(C=args.filter_C, max_iter=1000)
    linear_model.fit(clip_embeddings, df.to_numpy()[:, 1])

    # set threshold for person detection with bev heatmap
    bev_model.model.centermap_parser.conf_thresh = args.bev_detect_thresh
    # set the near/far thresholds for filtering out pose predictions (only keep predictions in middle range)
    bev_model.low_x_thresh = args.bev_low_x_thresh
    bev_model.high_x_thresh = args.bev_high_x_thresh
    bev_model.low_y_thresh = args.bev_low_y_thresh
    bev_model.high_y_thresh = args.bev_high_y_thresh
    bev_model.low_z_thresh = args.bev_low_z_thresh
    bev_model.high_z_thresh = args.bev_high_z_thresh

    print('Loading data...')
    laion_data = Laion(
                    args.laion_tar_files,
                    preprocess_type="bev_blip2",
                    min_orig_image_size=args.min_orig_image_size,
                )
    data_loader = laion_data.wds_loader(
        batch_size=args.batch_size,
        resampled=False,
        shuffle=0,
    )
    data_iterator = iter(data_loader)

    continue_collection = 1
    current_tarfile_id = 0
    sample_list = []

    while continue_collection:

        # load next batch using sentinel value None to indicate that dataloader is empty
        batch = next(data_iterator, None)

        if batch is None:
            # local variable to show dataloader is empty
            continue_collection = 0

        start_batch_time = time.time()
        if batch is not None:
            # remove samples with missing entries (given by "mask")
            mask = batch["mask"]    

            image_bev = batch["image_bev"][mask]
            image_blip2 = batch["image_blip2"][mask]
            image_clip_vit = batch["image_clip_vit"][mask]
            image_clip_res = batch["image_clip_res"][mask]
            image_open_clip = batch["image_open_clip"][mask]
            image_detectron2 = batch["image_detectron2"][mask]
            padding_bev = batch["padding_bev"][mask]

            text = np.array(batch["text"])[mask]
            pass_filter = np.array(batch["pass_filter"])[mask]
            key = np.array(batch["key"])[mask]
            url = np.array(batch["url"])[mask]

            image_bev = image_bev.to(device)
            image_blip2 = image_blip2.to(device)
            image_clip_vit = image_clip_vit.to(device)
            image_clip_res = image_clip_res.to(device)
            image_open_clip = image_open_clip.to(device)

            with torch.no_grad():

                # bev result for batch of images
                batch_outputs = bev_model(image_bev, padding_bev)

                # initial list of valid samples returned by bev, to be further filtered
                valid_status = np.array(
                    ['good' if outputs is not None else 'bev_no_person' for outputs in batch_outputs],
                    dtype='U255'
                )

                # post-bev sample filtering
                for img_idx in range(len(batch_outputs)):

                    if valid_status[img_idx] == 'good':
                        # remove laion samples using json info
                        if not pass_filter[img_idx]:
                            valid_status[img_idx] = 'json'

                    if valid_status[img_idx] == 'good':
                        # second person filter (after bev person filter via heatmap threshold)
                        person_detected = detect_person_detectron2(
                                            detectron2_model, image_detectron2[img_idx:(img_idx+1)]
                                        )
                        if not person_detected:
                            valid_status[img_idx] = 'detectron'

                    if valid_status[img_idx] == 'good':
                        # remove pose predictions which fall outside of image frame
                        batch_outputs_inframe = remove_out_of_frame_poses(
                                                    batch_outputs[img_idx],
                                                    padding_bev[img_idx].numpy(),
                                                    bev_model.smpl_parser,
                                                    completion_requirement=args.bev_completion_requirement,
                                                )
                        if batch_outputs_inframe is not None:
                            batch_outputs[img_idx] = batch_outputs_inframe
                        else:
                            valid_status[img_idx] = 'mesh'

                    if (not args.filter_phrases == "") and (valid_status[img_idx] == 'good'):
                        # remove poses from images which do not pass the hand-crafted clip filter
                        # TODO: batch-wise version of this?
                        # TODO: add support for open clip?
                        pass_filter_clip, _ = \
                            clip_filter(
                                image_clip=image_clip_vit[img_idx:(img_idx+1)],
                                clip_model=clip_model_vit,
                                filter_phrases=args.filter_phrases,
                                filter_phrase_thresholds=args.filter_phrase_thresholds,
                            )
                        if not pass_filter_clip:
                            valid_status[img_idx] = 'handcraft'

                    if valid_status[img_idx] == 'good':
                        # remove poses from images which do not pass the learned clip filter
                        # TODO: batch-wise version of this?
                        # TODO: add support for open clip?
                        pass_filter_open_clip = \
                            learned_clip_filter(
                                image_open_clip=image_open_clip[img_idx:(img_idx+1)],
                                open_clip_model=open_clip_model,
                                linear_model=linear_model,
                                threshold=args.filter_learned_threshold,
                            )
                        if not pass_filter_open_clip:
                            valid_status[img_idx] = 'learned'

                # caption only valid images
                image_blip2_caption = image_blip2[valid_status == 'good']
                image_clip_vit_caption = image_clip_vit[valid_status == 'good']
                image_clip_res_caption = image_clip_res[valid_status == 'good']
                if image_blip2_caption.shape[0] > 0:
                    blip2_text = caption_image_blip2(
                                        args=args,
                                        blip2_model=blip2_model,
                                        blip2_processor=blip2_processor,
                                        clip_model_vit=clip_model_vit,
                                        clip_model_res=clip_model_res,
                                        image_blip2=image_blip2_caption,
                                        image_clip_vit=image_clip_vit_caption,
                                        image_clip_res=image_clip_res_caption,
                                    )
                    if args.use_instruct_blip:

                        # get instruct blip caption
                        instruct_blip_text, pass_filter_instruct_blip = caption_image_instruct_blip(
                                            instruct_blip_model=instruct_blip_model,
                                            instruct_blip_processor=instruct_blip_processor,
                                            instruction_prompt=args.instruct_blip_prompt,
                                            images=image_blip2_caption,
                                            batch_size=args.instruct_blip_batch_size,
                                            max_length=args.instruct_blip_max_length,
                                            min_length=args.instruct_blip_min_length,
                                        )
                        # we noticed instruct blip gives short captions for images which are unsuitable for pose data,
                        # so we use the length of the instruct blip text as a final filter,
                        # and remove samples with short text
                        pass_instruct_blip_status = np.array([
                            'good' if pass_filt == 1
                            else 'instruct' for pass_filt in pass_filter_instruct_blip
                        ])
                        valid_status[valid_status == 'good'] = pass_instruct_blip_status
                        blip2_text = blip2_text[pass_instruct_blip_status == 'good']
                        instruct_blip_text = instruct_blip_text[pass_instruct_blip_status == 'good']

                # save results
                count = 0
                for img_idx in range(len(batch_outputs)):

                    if valid_status[img_idx] == 'good':

                        # get sample captions
                        sample_text = dict(blip_text=blip2_text[count], laion_text=text[img_idx])
                        if args.use_instruct_blip:
                            sample_text["instruct_blip_text"] = instruct_blip_text[count]
                        count += 1

                        # record sample info
                        sample = {
                            "__key__": key[img_idx],
                            "beta.npy": batch_outputs[img_idx]["smpl_betas"].cpu().numpy(),
                            "theta.npy": batch_outputs[img_idx]["smpl_thetas"].cpu().numpy(),
                            "trans.npy": batch_outputs[img_idx]["cam_trans"].cpu().numpy(),
                            "text.json": sample_text,
                        }
                        sample_list.append(sample)

                        # viz results
                        if args.render_selected:
                            assert world_size == 1, "Rendering only supported for single GPU currently."
                            # render using only data saved to tarfile
                            viz_text = (
                                "LAION Text: " + sample["text.json"]["laion_text"] + \
                                "\n" + "BLIP2 Text: " + sample["text.json"]["blip_text"]
                            )
                            if args.use_instruct_blip:
                                viz_text += (
                                    "\n" + "Instruct BLIP Text: " + \
                                    sample["text.json"]["instruct_blip_text"]
                                )
                            viz_render_bev(
                                thetas=sample["theta.npy"],
                                trans=sample["trans.npy"],
                                viz_text=viz_text,
                                key=sample["__key__"],
                                betas=sample["beta.npy"],
                                out_path=os.path.join(args.out_path, 'selected'),
                                laion_path=os.path.dirname(args.laion_tar_files),
                            )

                    if args.render_orig and batch_outputs[img_idx] is not None:
                        # visualize samples (sorted by validity status)
                        id_str = key[img_idx] + '_orig'
                        viz_image(
                            batch_outputs[img_idx]['rendered_image'][:, :, ::-1],
                            "",
                            os.path.join(args.out_path, valid_status[img_idx], f'im_{id_str}.png'),
                        )

        sample_list, current_tarfile_id = create_tarfiles(
            sample_list=sample_list,
            current_tarfile_id=current_tarfile_id,
            out_path=args.out_path,
            samples_per_tarfile=args.samples_per_tarfile,
            world_rank=world_rank,
            device=device,
            world_size=world_size,
            tarname_num_digits=args.tarname_num_digits,
        )
        if device == 0:
            print("\n")
        dist.barrier()

        # time and processing printout
        batch_run_time = time.time() - start_batch_time
        rate = args.batch_size / batch_run_time
        for worker_rank in range(world_size):
            if worker_rank == world_rank:
                current_laion_tarfile = max([
                    int(os.path.basename(tar_url).split('.')[0]) for tar_url in url
                ])
                print(
                    f"Device {world_rank} Processed {args.batch_size} images " + \
                    f"in {batch_run_time:.2f} seconds. " + \
                    f"Rate: {rate:.2f} images per second per GPU. " + \
                    f"Sample List Size: {len(sample_list)}. " + \
                    f"Currently scanning LAION tarfile: {current_laion_tarfile}"
                )
            dist.barrier()
        if device == 0:
            print("\n")

        # terminate collection when all dataloaders are empty
        continue_collection_gpu = torch.tensor([continue_collection]).long().to(device)
        continue_collection_gather = sync_tensor_across_workers(continue_collection_gpu)
        continue_collection = int(continue_collection_gather.max().cpu().numpy())

    # after collection, save remaining data to a final tarfile
    _, _ = create_tarfiles(
        sample_list=sample_list,
        current_tarfile_id=current_tarfile_id,
        out_path=args.out_path,
        samples_per_tarfile=max(args.min_samples_per_tarfile, len(sample_list)),
        world_rank=world_rank,
        device=device,
        world_size=world_size,
        tarname_num_digits=args.tarname_num_digits,
    )


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--laion_tar_files", type=str, default="/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data4/{00000..32449}.tar")
    parser.add_argument("--batch_size", type=int, default=200)
    # settings for saving sample tarfiles
    parser.add_argument("--out_path", type=str, default="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/testing/")
    parser.add_argument("--timestamp_out_path", type=lambda x: (str(x).lower() == 'true'), default=True, help="whether to add timestamp to save folder")
    parser.add_argument("--tarname_num_digits", type=int, default=5, help="number of digits in each tarfile name")
    parser.add_argument("--samples_per_tarfile", type=int, default=10000, help="number of pose samples to store in each output tarfile")
    parser.add_argument("--min_samples_per_tarfile", type=int, default=1000, help="min number of pose samples to store in partial output tarfiles")
    # bev settings
    parser.add_argument("--bev_detect_thresh", type=float, default=0.08, help="threshold for detecting person with bev heatmap")
    parser.add_argument("--bev_low_x_thresh", type=float, default=0.0, help="low threshold for depth filtering (remove poses below threshold)")
    parser.add_argument("--bev_high_x_thresh", type=float, default=120.0, help="high threshold for depth filtering (remove poses above threshold)")
    parser.add_argument("--bev_low_y_thresh", type=float, default=0.0, help="low threshold for depth filtering (remove poses below threshold)")
    parser.add_argument("--bev_high_y_thresh", type=float, default=120.0, help="high threshold for depth filtering (remove poses above threshold)")
    parser.add_argument("--bev_low_z_thresh", type=float, default=4.0, help="low threshold for depth filtering (remove poses below threshold)")
    parser.add_argument("--bev_high_z_thresh", type=float, default=50.0, help="high threshold for depth filtering (remove poses above threshold)")
    parser.add_argument("--bev_completion_requirement", type=float, default=0.85, help="proportion of pose mesh which must fall within image to keep pose sample")
    parser.add_argument("--render_selected", type=lambda x: (str(x).lower() == 'true'), default=True, help="render visualizations of saved data")
    parser.add_argument("--render_orig", type=lambda x: (str(x).lower() == 'true'), default=True, help="render visualizations from original bev pipeline (for comparison with viz from saved results)")
    # blip2/clip caption settings
    parser.add_argument("--blip2_reps", type=int, default=10, help="Number of BLIP-2 attempts for caption generation.")
    parser.add_argument("--blip2_max_length", type=int, default=30, help="Max token length of BLIP-2 caption.")
    parser.add_argument("--blip2_min_length", type=int, default=8, help="Min token length of BLIP-2 caption.")
    parser.add_argument("--blip2_round_1_size", type=int, default=3, help="Number of top ViT rankings to keep before final ranking.")
    parser.add_argument("--clip_len_penalty", type=float, default=0.075, help="Penalize CLIP logits by length of token sequence.")
    # instruct blip settings
    parser.add_argument("--use_instruct_blip", type=lambda x: (str(x).lower() == 'true'), default=True, help="whether to get caption with instruct blip")
    parser.add_argument("--instruct_blip_prompt", type=str, default="Describe the action and body position of the person or people.")
    parser.add_argument("--instruct_blip_batch_size", type=int, default=25, help="Max batch size for Instruct BLIP")
    parser.add_argument("--instruct_blip_max_length", type=int, default=77, help="Max token length for instruct blip captions")
    parser.add_argument("--instruct_blip_min_length", type=int, default=25, help="Min token length for instruct blip captions for sample inclusion. Used as filter, not to limit the output length of instruct blip")
    # hand-crafted filter settings
    parser.add_argument("--min_orig_image_size", type=int, default=200, help="minimum size of original LAION image to collect pose data")
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
    # learned filter settings
    parser.add_argument("--filter_dataset_path", type=str, default="filtering-23-10-13-20-03-53.csv")
    parser.add_argument("--filter_embeddings_path", type=str, default="clip_embeddings_23-10-13-20-03-53_v1.npy")
    parser.add_argument("--filter_C", type=float, default=4.0)
    parser.add_argument("--filter_learned_threshold", type=float, default=0.75)

    args = parser.parse_args()

    if args.timestamp_out_path:
        # add timestamp to output folder path
        args.out_path = os.path.join(args.out_path, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))

    # convert filter phrases/thresholds from comma-separated str to python list
    if not args.filter_phrases == "":
        args.filter_phrases = args.filter_phrases.split(",")
        args.filter_phrase_thresholds = [float(thresh) for thresh in args.filter_phrase_thresholds.split(",")]
        assert len(args.filter_phrases) == len(args.filter_phrase_thresholds)

    return args


if __name__ == "__main__":
    # rank info passed in through env
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    print('World Rank:   ', WORLD_RANK)
    print('Local Rank:   ', LOCAL_RANK)
    if WORLD_RANK == 0:
        print('World Size:   ', WORLD_SIZE)

    # args for running code
    args = setup_args()

    # dist setup
    torch.cuda.set_device(LOCAL_RANK)
    torch.cuda.empty_cache()
    dist.init_process_group('nccl', rank=WORLD_RANK, world_size=WORLD_SIZE)

    # save copy of code
    if WORLD_RANK == 0:
        out_path_no_end_slash = args.out_path[:-1] if args.out_path.endswith('/') else args.out_path
        code_folder = os.path.basename(out_path_no_end_slash) + "-code"
        code_out_path = os.path.join(os.path.dirname(out_path_no_end_slash), code_folder)
        os.makedirs(code_out_path)
        save_code_files(code_out_path)

    # run data collection
    collect_data(args, WORLD_RANK, LOCAL_RANK, WORLD_SIZE)

    # close dist
    dist.destroy_process_group()
