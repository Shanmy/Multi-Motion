import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
import argparse
import argparse
import time

import numpy as np
import trimesh
import smplx
import h5py
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../pose_datasets"))
from smplify import SMPLify3D
import config

from data_utils import get_pose_from_smpl_params

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1,
                    help='input batch size')
parser.add_argument('--num_smplify_iters', type=int, default=50,
                    help='num of smplify iters')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--gpu_ids', type=int, default=0,
                    help='choose gpu ids')
parser.add_argument('--num_joints', type=int, default=22,
                    help='joint number')
parser.add_argument('--joint_category', type=str, default="AMASS",
                    help='use correspondence')
parser.add_argument('--file_name', type=str, default="./demo/demo_data/whisper_4.npy",
                    help='data in the folder')
parser.add_argument('--save_dir', type=str, default="./demo/intergen/",
                    help='results save folder')
parser.add_argument('--width', type=int, default=512,
                    help='width of output image')
parser.add_argument('--height', type=int, default=512,
                    help='height of output image')
parser.add_argument('--exp', type=str, default="InterGen",
                    help='one of [InterGen, ComMDM, RIG]')
parser.add_argument('--mm', type=bool, default=False,
                    help='multi-modality evaluation')
opt = parser.parse_args()

def get_smpl_faces():
    return np.load(os.path.join(config.SMPL_MODEL_DIR, "smplfaces.npy"))

def save_pose_obj(vertices, save_path):
    mesh = trimesh.Trimesh(vertices=vertices.tolist(), faces=get_smpl_faces())
    with open(save_path, 'w') as fw:
        mesh.export(fw, 'obj')

if __name__ == "__main__":

    device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
    #load mean file as initial data
    file = h5py.File(config.SMPL_MEAN_FILE, 'r')
    init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).float()
    init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).float()
    
    smplmodel = smplx.create(config.SMPL_MODEL_DIR, 
                                model_type="smpl", gender="neutral", ext="pkl",
                                batch_size=opt.batchSize).to(device)
    # # #-------------initialize SMPLify
    smplify = SMPLify3D(smplxmodel=smplmodel,
                            batch_size=opt.batchSize,
                            joints_category=opt.joint_category,
                            num_iters=opt.num_smplify_iters,
                            device=device)
    
    total_num = 1024 if opt.mm else 2000
    
    for i in range(total_num):
        if not opt.mm:
            file_name = f"/nfs/USRCSEA/IVA/Datasets/motion_baselines/{opt.exp}/npy/{i:04d}.npy"
            if opt.exp == "RIG":
                file_name = f"/home/s9053168/code/Human-Interaction-Generation/codes/vis_data/gen_interaction/{i:04d}.npy"
            out_name = f"/nfs/USRCSEA/IVA/Datasets/motion_baselines/{opt.exp}/smpl/{i:04d}.npy"
        else:
            j = i % 20
            i = i // 20
            file_name = f"/nfs/USRCSEA/IVA/Datasets/motion_baselines/{opt.exp}/npy_mm/{i:04d}_{j:02d}.npy"
            out_name = f"/nfs/USRCSEA/IVA/Datasets/motion_baselines/{opt.exp}/smpl_mm/{i:04d}_{j:02d}.npy"
        if os.path.exists(out_name):
            continue
        data = np.load(file_name)  # [2 x 61 x 22 x 3]
    
        #2 person motion
        num_pers = data.shape[0]
        seq_len = data.shape[1]

        cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).to(device)
        pred_pose = torch.zeros(num_pers*data.shape[1], 72).to(device)
        pred_betas = torch.zeros(num_pers*data.shape[1], 10).to(device)
        pred_cam_t = torch.zeros(num_pers*data.shape[1], 3).to(device)
        keypoints_3d = torch.zeros(num_pers*data.shape[1], 22, 3).to(device)
        
        confidence_input =  torch.ones(22)
        # put more confidence into foot and ankle
        confidence_input[7] = 1.5
        confidence_input[8] = 1.5
        confidence_input[10] = 1.5
        confidence_input[11] = 1.5

        start = time.time()
        for idx in range(num_pers):
            for frm in range(seq_len):
                keypoints_3d[idx*seq_len+frm, :, :] = torch.Tensor(data[idx][frm]).to(device).float()
                pred_betas[idx*seq_len+frm, :] = init_mean_shape
                pred_pose[idx*seq_len+frm, :] = init_mean_pose
                pred_cam_t[idx*seq_len+frm, :] = cam_trans_zero

        # ----- from initial to fitting -------
        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
        new_opt_cam_t, new_opt_joint_loss = smplify(pred_pose.detach(),
                                                    pred_betas.detach(),
                                                    pred_cam_t.detach(),
                                                    keypoints_3d,
                                                    conf_3d=confidence_input.to(device),
                                                    seq_ind=0)
        
        full_betas = torch.concat((new_opt_betas.cpu(), torch.zeros(new_opt_betas.shape[0], 1)), axis=1)
        full_pose = get_pose_from_smpl_params(full_betas, new_opt_pose.cpu().detach(), new_opt_cam_t.cpu().detach(), align_intergen=True, convert_6d=True, process_trans=False)
        full_pose_1, full_pose_2 = full_pose[:seq_len], full_pose[seq_len:]
        full_pose_ours = np.concatenate([full_pose_1[:, None, ...], full_pose_2[:, None, ...]], axis=1)
        
        out_meshes = smplmodel(betas=new_opt_betas, global_orient=new_opt_pose[:, :3], body_pose=new_opt_pose[:, 3:], transl=new_opt_cam_t, return_verts=True)
        out_meshes = out_meshes['vertices'].cpu().detach().numpy()
        out_mesh_1, out_mesh_2 = out_meshes[:seq_len], out_meshes[seq_len:]
        out_mesh_ours = np.concatenate([out_mesh_1[:, None, ...], out_mesh_2[:, None, ...]], axis=1)
        print('{:.2f} second'.format(time.time()-start))
        
        np.save(out_name, full_pose_ours)
        print(out_name, full_pose_ours.shape)
        
        # out_mesh_ours = out_mesh_ours[:, :, :, [0, 2, 1]]
        # save_poses_from_verts(out_mesh_ours[::10], save_path_stem='demo', framerate=2)

    