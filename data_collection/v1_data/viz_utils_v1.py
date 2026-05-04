# visualize results from v1 data collection

import os
import argparse
from datetime import datetime

import matplotlib.pyplot as plt

import torch

from vis_human import setup_renderer, rendering_romp_bev_results
from bev.post_parser import SMPLA_parser, body_mesh_projection2image
from romp.utils import img_preprocess

import sys
sys.path.append('../')

from data_pose_v1 import LaionPose
from data_laion import load_img_tar


def viz_image_result(bev_output, caption, out_path, id=0, viz_height=30.0):
    # add caption title and visualize
    aspect_ratio = bev_output.shape[0] / bev_output.shape[1]
    fig = plt.figure(figsize=(viz_height * aspect_ratio, viz_height))
    ax = plt.axes(frameon=False, xticks=[],yticks=[])
    ax.imshow(bev_output)
    ax.set_title(caption)
    plt.savefig(os.path.join(out_path, 'test_im_{}.png'.format(id)),
                bbox_inches='tight', pad_inches=0)
    plt.close()


def viz_render(
    thetas,
    trans,
    viz_text,
    key,
    out_path,
    laion_path,
    betas=None,
    smpla_parser=None,
    smpl_path='/nfs/USRCSEA/IVA/Models/PoseCaptionData/ROMP_and_BEV/smpla_packed_info.pth',
    smil_path='/nfs/USRCSEA/IVA/Models/PoseCaptionData/ROMP_and_BEV/smil_packed_info.pth',
):
    if smpla_parser is None:
        smpla_parser = SMPLA_parser(smpl_path, smil_path).cuda()

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
    outputs = rendering_romp_bev_results(renderer, outputs, orig_image, rendering_cfgs)

    img = outputs["rendered_image"]
    img = img[:, :, [2, 1, 0]]  # BGR -> RGB
    viz_image_result(img, viz_text, out_path, id=key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_tar_files", type=str, default="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/v1/{00000..01596}.tar")
    parser.add_argument("--laion_data_folder", type=str, default="/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data2/")
    parser.add_argument("--burnin_batches", type=int, default=100)
    parser.add_argument("--out_path", type=str, default="out_laion_viz/")
    parser.add_argument("--num_samples_viz", type=int, default=50)
    args = parser.parse_args()

    # folder to save viz results
    out_path = os.path.join(args.out_path, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    os.makedirs(out_path, exist_ok=True)

    # pose parser
    smpl_path = '/nfs/USRCSEA/IVA/Models/PoseCaptionData/ROMP_and_BEV/smpla_packed_info.pth'
    smil_path = '/nfs/USRCSEA/IVA/Models/PoseCaptionData/ROMP_and_BEV/smil_packed_info.pth'
    smpla_parser = SMPLA_parser(smpl_path, smil_path).cuda()

    # make dataloader for wds pose data
    pose_dataset = LaionPose(urls=args.pose_tar_files)
    pose_dataloader = pose_dataset.wds_loader(
                            batch_size=args.num_samples_viz,
                            resampled=True,
                            yield_key=True,
                        )
    pose_iterator = iter(pose_dataloader)
    for _ in range(args.burnin_batches):
        _ = next(pose_iterator)

    # burin samples to shuffle better
    pose_samples = next(iter(pose_dataloader))
    thetas_samples = pose_samples[0]
    trans_samples = pose_samples[1]
    blip_text_samples = pose_samples[2]
    laion_text_samples = pose_samples[3]
    key = pose_samples[4]
    sample_iterator = zip(thetas_samples, trans_samples, blip_text_samples, laion_text_samples, key)

    for i, (thetas, trans, blip_text, laion_text, key) in enumerate(sample_iterator):
        print(f'Visualizing sample {i+1}.')
        viz_text = 'LAION Text: ' + laion_text + '\n' + 'BLIP Text: ' + blip_text
        viz_render(
            thetas,
            trans,
            viz_text,
            key,
            out_path=out_path,
            laion_path=args.laion_data_folder,
            smpla_parser=smpla_parser,
        )
