import torch

import sys
sys.path.append('../diffusion')
from models.pose2motion import Pose2Motion
from models.pose_model import PoseNet

batch_size = 10
max_len = 120
max_poses = 10
pose_feat = 158

pose_net = PoseNet().cuda()
pose_net_sd = pose_net.state_dict()
pose2motion_net = Pose2Motion().cuda()
pose2motion_net.load_state_dict(pose_net_sd, strict=False)

x = torch.randn([batch_size, max_len, max_poses, pose_feat]).cuda()
x_reshape = x.view(-1, max_poses, pose_feat).cuda()
timesteps = torch.zeros([max_len * batch_size]).long().cuda()
timesteps_joint = torch.zeros([batch_size]).long().cuda()

cond_dict = {}
cond_dict['num_poses'] = max_poses * torch.ones([max_len * batch_size]).long().cuda()
cond_dict['text'] = [''] * (max_len * batch_size)

cond_dict_joint = {}
cond_dict_joint['lengths'] = max_len * torch.ones([batch_size]).long().cuda()
cond_dict_joint['num_poses'] = max_poses * torch.ones([batch_size]).long().cuda()
cond_dict_joint['pose_cond'] = torch.randn([batch_size, max_poses, pose_feat]).cuda()
cond_dict_joint['pose_cond'] = torch.randn([batch_size, max_poses, pose_feat]).cuda()
cond_dict_joint['text'] = [''] * batch_size

x_out = pose2motion_net(x, timesteps_joint, **cond_dict_joint)
x_reshape_out = pose_net(x_reshape, timesteps, **cond_dict).view(batch_size, max_len, max_poses, pose_feat)

print((x_out - x_reshape_out).abs().max())
