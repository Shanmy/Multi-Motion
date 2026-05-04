import torch

import sys
sys.path.append("../diffusion/")
from models.pose_model import PoseNet
from models.pose2motion import Pose2Motion
from models.net_utils import (
    InputProcess,
    OutputProcess,
    PositionalEncoding,
    TimestepEmbedder,
    disable_layer_grad,
    join_pose_and_cond,
    separate_pose_and_cond,
    make_mask,
)


pose_net = PoseNet(
    cond_mask_prob=0.0,
    dropout=0.0,
    latent_dim=256,
    num_heads=4,
    use_text_condition=False,
    use_positional_encoding=True,
)
pose_net.load_state_dict(torch.load('/nfs/USRCSEA/IVA/Experiments/motion-diffusion/mitch-v1/motion-v1/motion_8_gpu/24-01-31-01-27-33/checkpoints/ema_0.9999_000001.pt'))

pose2motion_net = Pose2Motion(
    cond_mask_prob=0.0,
    dropout=0.0,
    latent_dim=256,
    num_heads=4,
    pad_pos_emb_start=False,
    skip_pose_layers=True,
    freeze_pose_model=True,
    freeze_pose_transformer_only=True,
    use_pose_positional_encoding=True,
    pose_model_ckpt=None,
)
pose2motion_net.seqTransEncoderMotion.load_state_dict(pose_net.seqTransEncoder.state_dict())
pose2motion_net.input_process.load_state_dict(pose_net.input_process.state_dict())
pose2motion_net.output_process.load_state_dict(pose_net.output_process.state_dict())
pose2motion_net.embed_timestep.load_state_dict(pose_net.embed_timestep.state_dict())

pose_net = pose_net.cuda()
pose_net.eval()

pose2motion_net = pose2motion_net.cuda()
pose2motion_net.eval()


batch_size = 10
num_frames = 120
num_poses = 10
pose_feat = 158

pose = torch.randn([batch_size, num_frames, pose_feat]).cuda()
timesteps = torch.zeros([batch_size]).long().cuda()
cond_dict = {
    'text': [''] * batch_size,
    'lengths': num_frames * torch.ones([batch_size]).long().cuda(),
}

pose_full = torch.zeros([batch_size, num_frames, num_poses, pose_feat]).cuda()
pose_full[:, :, 0] = pose
cond_dict_full = {
    'text': [''] * batch_size,
    'lengths': num_frames * torch.ones([batch_size]).long().cuda(),
    'num_poses': torch.ones([batch_size]).long().cuda(),
}

pose_out = pose_net(pose, timesteps, **cond_dict)
pose_full_out = pose2motion_net(pose_full, timesteps, **cond_dict_full)

pose_full_out_id_0 = pose_full_out[:, :, 0]
print('Test of full net: ', (pose_out - pose_full_out_id_0).abs().max())

print('Breakdown of net stages')
pose_out = pose_net.input_process(pose)
pose_full_out = pose2motion_net.input_process(pose_full)

pose_full_out_id_0 = pose_full_out[:, :, 0]
print('Input linear: ', (pose_out - pose_full_out_id_0).abs().max())

emb_motion, emb_pose = pose2motion_net._prepare_embs(
                                timesteps=timesteps,
                                text=cond_dict_full['text'],
                                pose_cond=torch.zeros([pose_full.shape[0], pose_full.shape[2], pose_full.shape[3]]).cuda(),
                                motion_reps=pose_full.shape[1],
                                pose_reps=pose_full.shape[2],
                            )
emb = pose_net._prepare_emb(timesteps=timesteps, text=cond_dict['text'])

emb_id_0 = emb_motion.view(batch_size, num_poses, 1, -1).permute(0, 2, 1, 3).contiguous()[:, :, :1]
print('Embeddings: ', (emb - emb_id_0).abs().max())

mask_full = make_mask(
                seq_lens=cond_dict_full['lengths'],
                max_len=pose_full.shape[1],
                device=pose_full.device,
                reps=pose_full.shape[2],
            )
seq_lens = (cond_dict['lengths'] if 'lengths' in cond_dict.keys() else cond_dict['num_poses'])
mask = make_mask(seq_lens=seq_lens, max_len=pose.shape[1], device=pose.device)

mask_id_0 = mask_full.view(batch_size, num_poses, 1, -1).permute(0, 2, 1, 3).contiguous()[:, :, :1]
print('Masks: ', (mask.float() - mask_id_0.float()).abs().max())
