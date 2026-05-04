# TODO: update this file???

import os
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from data import CustomDataset
from net_utils import import_pose_and_caption_nets
from processing_utils import detect_person_detectron2, caption_image
from viz_utils import viz_image


def collect_data(args):

    os.makedirs(args.out_path, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bev_model, blip2_model, blip2_processor, clip_model_vit, clip_model_res, detectron2_model = \
        import_pose_and_caption_nets(device)

    # set threshold for person detection with bev heatmap
    bev_model.model.centermap_parser.conf_thresh = args.bev_detect_thresh
    # set the near/far thresholds for filtering out pose predictions (only keep predictions in middle depth range)
    bev_model.low_x_thresh = 0
    bev_model.high_x_thresh = 128
    bev_model.low_y_thresh = 0
    bev_model.high_y_thresh = 128
    bev_model.low_z_thresh = args.bev_low_z_thresh
    bev_model.high_z_thresh = args.bev_high_z_thresh

    dataset = CustomDataset(args.dataset_path)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # list of pose and caption results for each image
    result = []
    ind_count = 0

    for idx, batch in enumerate(tqdm(data_loader)):
        if idx >= args.max_num_images:
            break

        image_bev, image_blip2, image_clip_vit, image_clip_res, padding, image_path = batch

        image_bev = image_bev.to(device)
        image_blip2 = image_blip2.to(device)
        image_clip_vit = image_clip_vit.to(device)
        image_clip_res = image_clip_res.to(device)

        with torch.no_grad():

            # bev model for mesh pose detection
            outputs_all = bev_model(image_bev, padding)

            # second person filter (after bev person filter via heatmap threshold)
            '''
            person_detected = detect_person_detectron2(detectron2_model, image_cv2)
            if not person_detected:
                print('No person detected! (detectron2)')
                continue
            '''

            # get caption for the image
            blip2_text = caption_image(args, blip2_model, blip2_processor, clip_model_vit,
                                       clip_model_res, image_blip2, image_clip_vit, image_clip_res)

            outputs_final = []
            for i, outputs in enumerate(outputs_all):
                if outputs is not None:

                    outputs['caption'] = blip2_text[i]
                    outputs['image_path'] = image_path[i]
                    outputs_final.append(outputs)

                    # viz results for single image
                    bev_output = outputs['rendered_image'][:, :, ::-1]
                    viz_image(bev_output, blip2_text[i], args.out_path, ind_count, args.viz_height)
                    ind_count += 1

        result.extend(outputs_final)

    # TODO: save result list in .pkl file?


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--dataset_path", type=str, default="/home/notebook/data/public/Dataset/overt/eb14f1a2f3fa526fa96111e13fb33f18r/ILSVRC2012/train/")
    parser.add_argument("--dataset_path", type=str, default="demo_data/frisbee")
    parser.add_argument("--out_path", type=str, default="out_image_test/")
    parser.add_argument("--max_num_images", type=int, default=500, help="Maximum number of images to test.")
    parser.add_argument("--viz_height", type=float, default=30.0, help="Height of result viz plots.")
    # bev settings
    parser.add_argument("--bev_detect_thresh", type=float, default=0.08, help="threshold for detecting person with bev heatmap")
    parser.add_argument("--bev_low_z_thresh", type=float, default=0.0, help="low threshold for depth filtering (remove poses below threshold)")
    parser.add_argument("--bev_high_z_thresh", type=float, default=64.0, help="high threshold for depth filtering (remove poses above threshold)")
    # blip2/clip caption settings
    parser.add_argument("--blip2_reps", type=int, default=10, help="Number of BLIP-2 attempts for caption generation.")
    parser.add_argument("--blip2_max_length", type=int, default=30, help="Max token length of BLIP-2 caption.")
    parser.add_argument("--blip2_min_length", type=int, default=8, help="Min token length of BLIP-2 caption.")
    parser.add_argument("--blip2_round_1_size", type=int, default=5, help="Number of top ViT rankings to keep before final ranking.")
    parser.add_argument("--clip_len_penalty", type=float, default=0.075, help="Penalize CLIP logits by length of token sequence.")
    args = parser.parse_args()
    # add timestamp to output folder path
    args.out_path = os.path.join(args.out_path, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))

    collect_data(args)


if __name__ == '__main__':
    main()
