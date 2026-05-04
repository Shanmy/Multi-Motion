import numpy as np

import torch
import torch.nn as nn
import clip

from .net_utils import (
    InputProcess,
    OutputProcess,
    PositionalEncoding,
    TimestepEmbedder,
    disable_layer_grad,
    join_pose_and_cond,
    separate_pose_and_cond,
    full_to_pose,
    pose_to_full,
    full_to_motion,
    motion_to_full,
    make_mask,
)
from .cfg_sampler import apply_text_cond_mask
from pose_datasets.data_utils import get_pose_pad_batch


class JointNet(nn.Module):
    def __init__(
        self,
        num_feat=158,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.0,
        activation="gelu",
        clip_dim=512,
        clip_version='ViT-B/32',
        cond_mask_prob=0.0,
        pad_pos_emb_start=False,
    ):
        super().__init__()

        self.num_feat = num_feat
        self.latent_dim = latent_dim 
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.clip_dim = clip_dim
        self.cond_mask_prob = cond_mask_prob
        self.pad_pos_emb_start = pad_pos_emb_start

        self.input_process = InputProcess(self.num_feat, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(
                                        self.latent_dim,
                                        dropout=self.dropout,
                                        pad_pos_emb_start=self.pad_pos_emb_start,
                                    )

        seqTransEncoderLayerMotion = nn.TransformerEncoderLayer(
                                            d_model=self.latent_dim,
                                            nhead=self.num_heads,
                                            dim_feedforward=self.ff_size,
                                            dropout=self.dropout,
                                            activation=self.activation,
                                            batch_first=True,
                                        )
        self.seqTransEncoderMotion = nn.TransformerEncoder(
                                        seqTransEncoderLayerMotion, num_layers=self.num_layers
                                    )
        seqTransEncoderLayerPose = nn.TransformerEncoderLayer(
                                            d_model=self.latent_dim,
                                            nhead=self.num_heads,
                                            dim_feedforward=self.ff_size,
                                            dropout=self.dropout,
                                            activation=self.activation,
                                            batch_first=True,
                                        )
        self.seqTransEncoder = nn.TransformerEncoder(
                                        seqTransEncoderLayerPose, num_layers=self.num_layers
                                    )

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        self.clip_version = clip_version
        self.load_and_freeze_clip(self.clip_version)

        self.output_process = OutputProcess(self.num_feat, self.latent_dim)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        # Must set jit=False for training
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',jit=False)
        clip.model.convert_weights(clip_model)
        self.clip_model = clip_model
        # Set in eval and freeze CLIP weights
        self.clip_model.eval()
        disable_layer_grad(self.clip_model)

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        # truncate beyond 77 tokens
        texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length]
        return self.clip_model.encode_text(texts).float()

    def _apply_motion_layer(self, x, emb_motion, motion_layer, motion_mask):
        batch_size = x.shape[0]
        x = full_to_motion(x)
        xseq = join_pose_and_cond(x, emb_motion)
        xseq = motion_layer(xseq, src_key_padding_mask=motion_mask)
        x, emb_motion = separate_pose_and_cond(xseq)
        x = motion_to_full(x, batch_size)
        return x, emb_motion

    def _apply_pose_layer(self, x, emb_pose, pose_layer, pose_mask):
        batch_size = x.shape[0]
        x = full_to_pose(x)
        xseq = join_pose_and_cond(x, emb_pose)
        xseq = pose_layer(xseq, src_key_padding_mask=pose_mask)
        x, emb_pose = separate_pose_and_cond(xseq)
        x = pose_to_full(x, batch_size)
        return x, emb_pose

    def _apply_layer_pair(
        self,
        x,
        emb_motion,
        emb_pose,
        motion_layer,
        motion_mask,
        pose_layer,
        pose_mask,
    ):
        x, emb_pose = self._apply_pose_layer(x, emb_pose, pose_layer, pose_mask)
        x, emb_motion = self._apply_motion_layer(x, emb_motion, motion_layer, motion_mask)
        return x, emb_motion, emb_pose

    def _prepare_embs(self, timesteps, text, motion_reps, pose_reps):
        emb_time = self.embed_timestep(timesteps)  # [batch_size, model_feat]
        enc_text = self.encode_text(text) # [batch_size, 512]
        emb_text = self.embed_text(enc_text)  # [batch_size, model_feat]
        emb = emb_time + emb_text  # [batch_size, model_feat]

        # [batch_size, model_feat] -> [batch_size * poses, 1, model_feat]
        emb_motion = emb.unsqueeze(1).repeat(1, pose_reps, 1)
        emb_motion = emb_motion.view(emb.shape[0] * pose_reps, 1, -1)
        # [batch_size, model_feat] -> [batch_size * frames, 1, model_feat]
        emb_pose = emb.unsqueeze(1).repeat(1, motion_reps, 1)
        emb_pose = emb_pose.view(emb.shape[0] * motion_reps, 1, -1)

        return emb_motion, emb_pose

    def _apply_motion_positional_encoding(self, x, emb_motion, seq_lens):
        batch_size = x.shape[0]
        # adding the timestep embed to frame only. put pose on batch dim
        x = full_to_motion(x)  # [batch_size*poses, frames, model_feat]
        xseq = join_pose_and_cond(x, emb_motion)  # [batch_size*poses, frames+1, model_feat]
        xseq = self.sequence_pos_encoder(xseq, seq_lens=seq_lens)  # [batch_size*poses, frames+1 model_feat]
        x, emb_motion = separate_pose_and_cond(xseq)
        x = motion_to_full(x, batch_size)  # [batch_size, frames, poses, model_feat]
        return x, emb_motion

    def forward(self, x, timesteps, **cond_dict):
        """
        x: [batch_size, num_frames, num_person, pose_feat]
        timesteps: [batch_size] (int)
        cond_dict: dict with keys:
            'text': list of raw texts for each batch sample
            'lengths': [batch_size], gives number of motion frames
            'pose_mask': [batch_size], gives number of poses
        """

        # conditions and masks
        if self.training and self.cond_mask_prob > 0:
            cond_dict['text'] = apply_text_cond_mask(cond_dict['text'], self.cond_mask_prob)
        emb_motion, emb_pose = self._prepare_embs(
                                        timesteps=timesteps,
                                        text=cond_dict['text'],
                                        motion_reps=x.shape[1],
                                        pose_reps=x.shape[2],
                                    )
        motion_mask = make_mask(
                        seq_lens=cond_dict['lengths'],
                        max_len=x.shape[1],
                        device=x.device,
                        reps=x.shape[2],
                    )
        pose_mask = make_mask(
                        seq_lens=cond_dict['num_poses'],
                        max_len=x.shape[2],
                        device=x.device,
                        reps=x.shape[1],
                    )
        max_len = x.shape[1]
        max_poses = x.shape[2]

        # input projection and motion positional embedding. x shape: [batch_size, frames, poses, pose_feat]
        x = self.input_process(x)
        x, emb_motion = self._apply_motion_positional_encoding(x, emb_motion, seq_lens=cond_dict['lengths'])

        # sequence of (motion layer, pose layer)
        for idx in range(len(self.seqTransEncoderMotion.layers)):
            x, emb_motion, emb_pose = self._apply_layer_pair(
                x=x, 
                emb_motion=emb_motion,
                emb_pose=emb_pose,
                motion_layer=self.seqTransEncoderMotion.layers[idx],
                motion_mask=motion_mask,
                pose_layer=self.seqTransEncoder.layers[idx],
                pose_mask=pose_mask,
            )  # x and emb keep the same shape

        output = self.output_process(x)  # [batch_size, frames, poses, pose_feat]
        if not self.training:
            # pad output in eval mode
            output = get_pose_pad_batch(
                        output,
                        lengths=cond_dict['lengths'],
                        num_poses=cond_dict['num_poses'],
                        max_len=max_len,
                        max_poses=max_poses,
                    )
        return output


if __name__ == "__main__":

    from pose_datasets.data_utils import set_dummy_poses_to_zero

    joint_net = JointNet(dropout=0.0).cuda().eval()

    batch_size = 2
    max_len = 120
    max_poses =10
    pose_feat = 158

    # test masking
    test_samples = []
    # same as test samples for mask=True samples, but different for mask=False samples
    test_samples_mask_perturbed = []
    # masks to test output diffs after model fwd
    mask = []
    cond_dict = {
        'text': [''] * batch_size,
        'lengths': [],
        'num_poses': [],
    }
    for i in range(batch_size):
        test_sample = torch.randn([max_len, max_poses, pose_feat]).cuda()

        mask_len_loc = torch.randint(size=[1], high=max_len)
        mask_pose_loc = torch.randint(size=[1], high=max_poses)
        mask = torch.zeros([max_len, max_poses, 1]).cuda()
        mask[mask_len_loc:, mask_pose_loc:] = 1

        test_sample_mask_perturbed = test_sample.clone()
        test_sample_mask_perturbed += mask * torch.randn_like(test_sample)
        # line below should cause test to fail
        #test_sample_mask_perturbed += torch.randn_like(test_sample)

        test_samples.append(test_sample)
        test_samples_mask_perturbed.append(test_sample_mask_perturbed)

        cond_dict['lengths'].append(int(mask_len_loc))
        cond_dict['num_poses'].append(int(mask_pose_loc))
    cond_dict['lengths'] = torch.tensor(cond_dict['lengths']).cuda()
    cond_dict['num_poses'] = torch.tensor(cond_dict['num_poses']).cuda()

    test_samples = torch.stack(test_samples)
    test_samples_mask_perturbed = torch.stack(test_samples_mask_perturbed)
    timesteps = torch.randint(size=[batch_size], high=1000).long().cuda()

    with torch.no_grad():
        out = joint_net(test_samples, timesteps, **cond_dict)
        out_perturbed = joint_net(test_samples_mask_perturbed, timesteps, **cond_dict)

    diff_sq_unmasked = (out - out_perturbed) ** 2
    diff_sq = set_dummy_poses_to_zero(
        diff_sq_unmasked, lengths=cond_dict['lengths'], num_poses=cond_dict['num_poses']
    )
    print('Test masking. If successful, output should be close to 0: ', diff_sq.abs().max())
