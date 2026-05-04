# visualize results from data collection

import os
import argparse
from datetime import datetime

from PIL import Image

import torch

from vis_human import setup_renderer, rendering_romp_bev_results
from bev.post_parser import body_mesh_projection2image
from romp.utils import img_preprocess

from viz_utils import viz_image

from pose_datasets.laion_pose_loader import LaionPoseWDS
from pose_datasets.data_utils import remove_pad_poses
from data_collection.data_laion import load_img_tar
from smpl.smpl import SMPLA_parser


# might be needed in mesh editing code
def get_padding_bev(image):
    image = Image.fromarray(image)
    w, h = image.size
    side_length = max(w, h)
    top, left = int((side_length - h) // 2), int((side_length - w) // 2)
    bottom, right = int(top+h), int(left+w)
    image_pad_info = [top, bottom, left, right, h, w]
    return image_pad_info


def viz_render_bev(
    betas,  # set to None for default betas
    thetas,
    trans,
    viz_text,
    key,
    out_path,
    laion_path,
    smpla_parser=None,
    id_str='test_im',
    num_poses=None,
):
    if smpla_parser is None:
        smpla_parser = SMPLA_parser().cuda()

    if num_poses is not None:
        # remove pad poses
        betas = remove_pad_poses(betas, num_frames=None, num_poses=num_poses)
        thetas = remove_pad_poses(thetas, num_frames=None, num_poses=num_poses)
        trans = remove_pad_poses(trans, num_frames=None, num_poses=num_poses)

    if not torch.is_tensor(thetas):
        thetas = torch.tensor(thetas)

    if betas is None:
        # default value for betas (neutral shape). if betas are neutral shape, viz will not exactly match original viz
        betas = torch.zeros(size=[thetas.shape[0], 11])
    elif not torch.is_tensor(betas):
        betas = torch.tensor(betas)

    if not torch.is_tensor(trans):
        trans = torch.tensor(trans)

    num_people = len(thetas.abs().max(dim=1)[0].nonzero()[:, 0])
    thetas = thetas[:num_people].cuda()
    betas = betas[:num_people].cuda()
    trans = trans[:num_people].cuda()

    # setup renderer
    renderer = setup_renderer(name='sim3dr')
    rendering_cfgs = {'mesh_color': 'identity', 'items': ['mesh', 'mesh_bird_view'], 'renderer': 'sim3dr'}

    # load original image
    orig_image = load_img_tar(key, laion_path=laion_path)
    _, image_pad_info = img_preprocess(orig_image)

    # process smpl and translation params
    outputs = {'smpl_thetas': thetas, 'smpl_betas': betas, 'cam_trans': trans}
    verts, joints, face = smpla_parser(outputs['smpl_betas'], outputs['smpl_thetas']) 
    outputs.update({'verts': verts, 'joints': joints, 'smpl_face': face})
    outputs.update(
        body_mesh_projection2image(
            outputs['joints'],
            outputs['cam_trans'],
            vertices=outputs['verts'],
            input2org_offsets=image_pad_info,
            denormalize_trans=False,
        )
    )
    outputs = rendering_romp_bev_results(renderer, outputs, orig_image[:, :, [2, 1, 0]], rendering_cfgs)

    img = outputs["rendered_image"][:, :, [2, 1, 0]]
    viz_image(img, viz_text, os.path.join(out_path, f'{id_str}_key_{key}.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_tar_files", type=str, default="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/v2/{00000..00300}.tar")
    parser.add_argument("--laion_data_folder", type=str, default="/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data4/")
    parser.add_argument("--burnin_batches", type=int, default=1000)
    parser.add_argument("--out_path", type=str, default="out_viz_bev/")
    parser.add_argument("--num_samples_viz", type=int, default=50)
    args = parser.parse_args()
    
    # 000060022
    import cv2
    key = ["000064208", "000060022", "000060106", "000060164", "000060308", "000061303", "000061402", "000062044", "000062400", "000062893", "000063109", "000063648"]
    for k in key:
        orig_image = load_img_tar(k, laion_path=args.laion_data_folder)
        cv2.imwrite(f"/home/s9053168/code/multi-pose-diffusion/viz/laion_data_orig/{k}.png", orig_image[:, :, [2, 1, 0]])

    # folder to save viz results
    out_path = os.path.join(args.out_path, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    os.makedirs(out_path, exist_ok=True)

    # make dataloader for wds pose data
    pose_dataset = LaionPoseWDS(urls=args.pose_tar_files)
    pose_dataloader = pose_dataset.wds_loader(
                            batch_size=args.num_samples_viz,
                            resampled=True,
                            concat_vars=False,
                            yield_cond_dict=False,
                            yield_key=True,
                        )
    # burnin samples to shuffle better
    pose_iterator = iter(pose_dataloader)
    for _ in range(args.burnin_batches):
        _ = next(pose_iterator)

    # get batch for viz
    pose_samples = next(iter(pose_dataloader))
    betas_samples = pose_samples[0]
    thetas_samples = pose_samples[1]
    trans_samples = pose_samples[2]
    laion_text_samples = pose_samples[3]
    blip_text_samples = pose_samples[4]
    instruct_blip_text_samples = pose_samples[5]
    keys = pose_samples[6]
    sample_iterator = zip(
        betas_samples,
        thetas_samples,
        trans_samples,
        blip_text_samples,
        laion_text_samples,
        keys
    )

    for i, (betas, thetas, trans, blip_text, laion_text, key) in enumerate(sample_iterator):
        print(f'Visualizing sample {i+1}.')
        viz_text = (
            'LAION Text: ' + laion_text + '\n' +
            'BLIP Text: ' + blip_text + '\n' +
            'Instruct BLIP Text: ' + instruct_blip_text_samples[i]
        )
        viz_render_bev(
            betas,
            thetas,
            trans,
            viz_text,
            key,
            out_path=out_path,
            laion_path=args.laion_data_folder,
        )
