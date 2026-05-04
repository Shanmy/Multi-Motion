import argparse

from smpl.smpl import SMPLA_parser
from pose_datasets.laion_pose_loader import LaionPoseWDS
from pose_datasets.data_utils import get_pose_from_smpl_params
from data_collection.data_laion import load_img_tar
from viz.viz_bev import get_padding_bev

from bev.post_parser import body_mesh_projection2image


def calculate_proportion_in_frame(
    betas,
    thetas,
    trans,
    padding_bev,
    smpla_parser=None,
):
    # order of padding_bev: [top_pad, bottom_pad, left_pad, right_pad, height, width]

    # load smpl models
    if smpla_parser is None:
        smpla_parser = SMPLA_parser().to(betas.device)

    # get estimated meshes
    verts, joints, face = smpla_parser(betas, thetas)
    outputs = {}
    outputs.update(
        body_mesh_projection2image(
            joints,
            trans,
            vertices=verts,
            input2org_offsets=padding_bev,
            denormalize_trans=False,
        )
    )

    # key 'verts_camed_org' are the vertices we want
    for i, vertex_sample in enumerate(outputs['verts_camed_org'].cpu().numpy()):

        # remove pose predictions where a large proportion of mesh falls outside of image
        vertical_filter = (vertex_sample[:, 0] < 0.0) | (vertex_sample[:, 0] > padding_bev[5])
        horizontal_filter = (vertex_sample[:, 1] < 0.0) | (vertex_sample[:, 1] > padding_bev[4])
        boundary_filter = horizontal_filter | vertical_filter
        # printouts below should be >=0.85 for laion pose v2 data (already selected according to that criteria)
        print(f'DEMO | Mesh proportion in frame: {1 - (boundary_filter.sum() / vertex_sample.shape[0])}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_tar_files", type=str, default="/nfs/USRCSEA/IVA/Datasets/MotionDiff/Text2Pose/laion_pose/v2/{00000..00815}.tar")
    parser.add_argument("--laion_data_folder", type=str, default="/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data4/")
    parser.add_argument("--num_samples_viz", type=int, default=50)
    args = parser.parse_args()

    # make dataloader for wds pose data
    pose_dataset = LaionPoseWDS(urls=args.pose_tar_files)
    pose_dataloader = pose_dataset.wds_loader(
                            batch_size=args.num_samples_viz,
                            resampled=True,
                            concat_vars=False,
                            yield_cond_dict=False,
                            yield_num_poses=True,
                            yield_key=True,
                        )
    # burnin samples to shuffle better
    pose_iterator = iter(pose_dataloader)

    # get batch for viz
    pose_samples = next(iter(pose_dataloader))
    betas_samples = pose_samples[0]
    thetas_samples = pose_samples[1]
    trans_samples = pose_samples[2]
    num_poses_samples = pose_samples[-2]
    keys = pose_samples[-1]
    sample_iterator = zip(betas_samples, thetas_samples, trans_samples, num_poses_samples, keys)

    for i, (betas, thetas, trans, num_poses, key) in enumerate(sample_iterator):
        orig_image = load_img_tar(key, laion_path=args.laion_data_folder)
        padding_bev = get_padding_bev(orig_image)
        betas = betas[0:num_poses].cuda()
        thetas = thetas[0:num_poses].cuda()
        trans = trans[0:num_poses].cuda()
        calculate_proportion_in_frame(
            betas=betas,
            thetas=thetas,
            trans=trans,
            padding_bev=padding_bev,
        )
        # demo to convert to 158-dim (non-amass-aligned) pose vector
        pose = get_pose_from_smpl_params(betas.cpu(), thetas.cpu(), trans.cpu())
