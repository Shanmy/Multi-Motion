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
    zero_module,
)
from .cfg_sampler import apply_text_cond_mask
from pose_datasets.data_utils import get_pose_pad_batch, process_translations


# condition first frame of motion on ground-truth pose.

class Pose2Motion(nn.Module):
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
        use_pose_positional_encoding=True,
        use_pose_residual_format=True,
        pad_pos_emb_start=False,
        pose_model_ckpt=None,
        freeze_pose_model=False,
        freeze_pose_transformer_only=False,
        freeze_motion_model=False,
        skip_pose_layers=False,
        skip_motion_layers=False,
        use_text_condition=True,
        use_pose_condition=False,
        use_pose_condition_first_frame=True,
        center_pose_cond=True,
        use_pose_only_first_frame=True,
        inpaint_gt_first_pose=False,
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
        self.use_pose_positional_encoding = use_pose_positional_encoding
        self.use_pose_residual_format = use_pose_residual_format
        self.pad_pos_emb_start = pad_pos_emb_start
        self.use_text_condition = use_text_condition
        self.use_pose_condition = use_pose_condition
        self.use_pose_condition_first_frame = use_pose_condition_first_frame
        self.center_pose_cond = center_pose_cond
        self.use_pose_only_first_frame = use_pose_only_first_frame

        self.freeze_pose_model = freeze_pose_model
        self.freeze_pose_transformer_only = freeze_pose_transformer_only
        self.freeze_motion_model = freeze_motion_model
        self.skip_pose_layers = skip_pose_layers
        self.skip_motion_layers = skip_motion_layers

        # during inference time, whether first pose comes from data or model predictions (2 stage only)
        self.inpaint_gt_first_pose = inpaint_gt_first_pose

        self.input_process = InputProcess(self.num_feat, self.latent_dim)

        self.sequence_pos_encoder_motion = PositionalEncoding(
                                        self.latent_dim,
                                        dropout=self.dropout,
                                        pad_pos_emb_start=self.pad_pos_emb_start,
                                    )
        if use_pose_positional_encoding:
            self.sequence_pos_encoder_pose = PositionalEncoding(
                                            self.latent_dim,
                                            dropout=self.dropout,
                                            pad_pos_emb_start=False,
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
        for idx in range(len(self.seqTransEncoderMotion.layers)):
            zero_module(self.seqTransEncoderMotion.layers[idx].norm2)

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
        for idx in range(len(self.seqTransEncoder.layers)):
            if self.use_pose_residual_format:
                zero_module(self.seqTransEncoder.layers[idx].norm2)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder_motion)

        if self.use_pose_condition:
            self.embed_pose_cond = nn.Sequential(
                                    nn.Linear(2 * self.num_feat, self.latent_dim),
                                    nn.GELU(),
                                    nn.Linear(self.latent_dim, self.latent_dim),
                                )

        if self.use_text_condition:
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            self.clip_version = clip_version
            self.load_and_freeze_clip(self.clip_version)

        self.output_process = OutputProcess(self.num_feat, self.latent_dim)

        self.pose_model_ckpt = pose_model_ckpt
        if self.pose_model_ckpt is not None:
            self.load_state_dict(torch.load(self.pose_model_ckpt, map_location='cpu'), strict=False)

        if self.freeze_pose_model:
            self.freeze_pose_model_parameters(freeze_all=(not self.freeze_pose_transformer_only))
        if self.freeze_motion_model:
            self.freeze_motion_model_parameters()

    def _center_cond_pose(self, poses, num_poses):
        assert len(poses.shape) == 2
        # hacky handling of device
        # done to keep process_translations as a numpy-only fn for now...
        if self.center_pose_cond:
            trans = poses[:num_poses, -3:].cpu().numpy()
            trans = process_translations(
                                        trans,
                                        trans_aug_rad=None,
                                        single_motion_mode=False,
                                        center_first_frame=True,
                                        center_trans=True,
                                    )
            poses[:num_poses, -3:] = torch.tensor(trans).to(poses.device)
        return poses

    def update_cond_dict(self, batch, cond_dict):
        if self.use_pose_condition:
            mask_bools = [
                torch.ones_like(batch[idx, 0]) if cond_dict['lengths'][idx] > 1
                else torch.zeros_like(batch[idx, 0])
                for idx in range(batch.shape[0])
            ]
            # TODO: condition augmentation?
            # first pose is condition, except for 1-frame pose type data which gets null condition
            if self.use_pose_condition_first_frame:
                # first frame condition
                cond_idx = torch.zeros([batch.shape[0]]).long().to(batch.device)
            else:
                if batch.shape[1] == 1:
                    # special case for single-frame condition from stage 1 model
                    cond_idx = torch.zeros([batch.shape[0]]).long().to(batch.device)
                else:
                    # middle frame condition (training or sampling initalized from data sample middle frame)
                    cond_idx = cond_dict['lengths'] // 2

            cond_pose = torch.zeros([batch.shape[0], batch.shape[2], batch.shape[3]]).to(batch.device)
            for idx in range(batch.shape[0]):
                cond_pose[idx] = self._center_cond_pose(
                                    batch[idx, cond_idx[idx]],
                                    cond_dict['num_poses'][idx]
                                )

            cond_dict['pose_cond'] = torch.stack([
                torch.cat((cond_pose[idx], mask_bools[idx]), -1)
                if cond_dict['lengths'][idx] > 1
                else torch.cat((torch.zeros_like(batch[idx, 0]), mask_bools[idx]), -1)
                for idx in range(batch.shape[0])
            ])
        return cond_dict

    def denoised_fn(self, batch, cond_dict):
        if self.use_pose_condition and self.inpaint_gt_first_pose:
            if self.use_pose_condition_first_frame:
                # first frame condition
                cond_idx = torch.zeros([batch.shape[0]]).long().to(batch.device)
            else:
                # middle frame condition
                cond_idx = cond_dict['lengths'] // 2
            # overwrite model denoised predictions with ground truth first pose
            for idx in range(batch.shape[0]):
                batch[idx, cond_idx[idx]] = cond_dict['pose_cond'][idx, :, :batch.shape[-1]]
        return batch

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

    def freeze_pose_model_parameters(self, freeze_all=False):
        disable_layer_grad(self.seqTransEncoder)
        if self.use_text_condition:
            disable_layer_grad(self.embed_text)
        if freeze_all:
            disable_layer_grad(self.input_process)
            disable_layer_grad(self.embed_timestep)
            disable_layer_grad(self.output_process)

    def freeze_motion_model_parameters(self):
        disable_layer_grad(self.seqTransEncoderMotion)
        if self.use_pose_condition:
            disable_layer_grad(self.embed_pose_cond)

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        # truncate beyond 77 tokens
        texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length]
        return self.clip_model.encode_text(texts).float()

    def _apply_motion_layer(self, x, emb_motion, motion_layer, motion_mask):
        batch_size = x.shape[0]

        if self.use_pose_only_first_frame:
            # copy of first pose to append after motion layer (first pose only goes through pose layers)
            _, x_pose = separate_pose_and_cond(x)  # takes away first token x_pose

        # process motion layers in framewise attention format
        x = full_to_motion(x)
        xseq = join_pose_and_cond(x, emb_motion)

        xseq_res = self.sequence_pos_encoder_motion(xseq)
        xseq_res = motion_layer(xseq_res, src_key_padding_mask=motion_mask)
        xseq = xseq + xseq_res

        x, emb_motion = separate_pose_and_cond(xseq)
        x = motion_to_full(x, batch_size)

        if self.use_pose_only_first_frame:
            # remove layer outputs for the 2nd to final frame
            x_motion, _ = separate_pose_and_cond(x)
            # join input first frame feats with transformer outputs for other frames
            x = join_pose_and_cond(x_motion, x_pose)  # adds x_pose to first token

        return x, emb_motion

    def _apply_pose_layer(self, x, emb_pose, pose_layer, pose_mask):
        batch_size = x.shape[0]
        x = full_to_pose(x)
        xseq = join_pose_and_cond(x, emb_pose)

        if self.use_pose_residual_format:
            xseq_res = xseq
            if self.use_pose_positional_encoding:
                xseq_res = self.sequence_pos_encoder_pose(xseq_res)
            xseq_res = pose_layer(xseq_res, src_key_padding_mask=pose_mask)
            xseq = xseq + xseq_res
        else:
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
        if not self.skip_pose_layers:
            x, emb_pose = self._apply_pose_layer(x, emb_pose, pose_layer, pose_mask)
        if not self.skip_motion_layers:
            x, emb_motion = self._apply_motion_layer(x, emb_motion, motion_layer, motion_mask)
        return x, emb_motion, emb_pose

    def _prepare_embs(
        self,
        timesteps,
        motion_reps,
        pose_reps,
        text=None,
        text_encoding=None,
        pose_cond=None,
    ):
        batch_size = timesteps.shape[0]

        emb_time = self.embed_timestep(timesteps)  # [batch_size, model_feat]

        if self.use_text_condition:
            assert text is not None or text_encoding is not None
            if text_encoding is None:
                enc_text = self.encode_text(text) # [batch_size, 512]
            else:
                # precomputed encoding, for faster test-time sampling
                enc_text = text_encoding
            emb_text = self.embed_text(enc_text)  # [batch_size, model_feat]

        if self.use_pose_condition:
            emb_pose_cond = self.embed_pose_cond(pose_cond)  # [batch_size, max_poses, model_feat]

        # time (and optional text condition) for pose layer (inhereted from base model)
        if self.use_text_condition:
            emb_pose = emb_time + emb_text  # [batch_size, model_feat]
        else:
            emb_pose = emb_time  # [batch_size, model_feat]
        # [batch_size, model_feat] -> [batch_size * frames, 1, model_feat]
        emb_pose = emb_pose.unsqueeze(1).repeat(1, motion_reps, 1)
        emb_pose = emb_pose.view(batch_size * motion_reps, 1, -1)

        # time (and optional first frame pose condition) for motion layers
        emb_motion = emb_time  # [batch_size, model_feat]
        # [batch_size, model_feat] -> [batch_size * poses, 1, model_feat]
        emb_motion = emb_motion.unsqueeze(1).repeat(1, pose_reps, 1)
        if self.use_pose_condition:
            # each invididual motion conditioned on first frame of that sequence
            emb_motion = emb_motion + emb_pose_cond
        emb_motion = emb_motion.view(batch_size * pose_reps, 1, -1)

        return emb_motion, emb_pose

    def _apply_pose_positional_encoding(self, x, emb_pose):
        batch_size = x.shape[0]
        x = full_to_pose(x)
        xseq = join_pose_and_cond(x, emb_pose)
        xseq = self.sequence_pos_encoder_pose(xseq)
        x, emb_pose = separate_pose_and_cond(xseq)
        x = pose_to_full(x, batch_size)
        return x, emb_pose

    def forward(self, x, timesteps, **cond_dict):
        """
        x: [batch_size, max_num_frames, max_num_poses, pose_feat]
        timesteps: [batch_size] (int)
        cond_dict: dict with keys:
            'text': list of raw texts for each batch sample
            'lengths': [batch_size], gives number of motion frames
            'num_poses': [batch_size], gives number of poses
            'pose_cond': [batch_size, num_poses, 158]
        """

        # code below for debug only. TODO: add classifier-free guidance type method for here?
        #if not 'pose_cond' in cond_dict.keys():
        #    cond_dict['pose_cond'] = torch.zeros([x.shape[0], x.shape[2], x.shape[3]]).to(x.device)

        # conditions and masks
        if self.training and self.cond_mask_prob > 0:
            cond_dict['text'] = apply_text_cond_mask(cond_dict['text'], self.cond_mask_prob)
        emb_motion, emb_pose = self._prepare_embs(
            timesteps=timesteps,
            motion_reps=x.shape[1],
            pose_reps=x.shape[2],
            text=(cond_dict['text'] if 'text' in cond_dict.keys() else None),
            text_encoding=(cond_dict['text_encoding'] if 'text_encoding' in cond_dict.keys() else None),
            pose_cond=(cond_dict['pose_cond'] if self.use_pose_condition else None),
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

        if self.use_pose_positional_encoding and not self.use_pose_residual_format:
            x, emb_pose = self._apply_pose_positional_encoding(x, emb_pose)

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

    joint_net = Pose2Motion(dropout=0.0).cuda().eval()

    batch_size = 2
    max_len = 120
    max_poses = 10
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
    cond_dict['pose_cond'] = torch.randn([batch_size, max_poses, pose_feat]).cuda()

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
