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
    make_mask,
    zero_module,
)
from .cfg_sampler import apply_text_cond_mask
from pose_datasets.data_utils import get_pose_pad_batch


class PoseNet(nn.Module):
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
        use_positional_encoding=True,
        use_text_condition=True,
        use_pose_condition=False,
        use_pose_condition_first_frame=True,
        use_layer_residual_format=True,
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
        self.use_positional_encoding = use_positional_encoding
        self.use_text_condition = use_text_condition
        self.use_pose_condition = use_pose_condition
        self.use_pose_condition_first_frame = use_pose_condition_first_frame
        self.use_layer_residual_format = use_layer_residual_format

        # during inference time, whether first pose comes from data or model predictions (2 stage only)
        self.inpaint_gt_first_pose = inpaint_gt_first_pose

        self.input_process = InputProcess(self.num_feat, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(
                                        self.latent_dim,
                                        dropout=self.dropout,
                                        pad_pos_emb_start=False
                                    )

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
                                            d_model=self.latent_dim,
                                            nhead=self.num_heads,
                                            dim_feedforward=self.ff_size,
                                            dropout=self.dropout,
                                            activation=self.activation,
                                            batch_first=True,
                                        )
        self.seqTransEncoder = nn.TransformerEncoder(
                                        seqTransEncoderLayer, num_layers=self.num_layers
                                    )
        for idx in range(len(self.seqTransEncoder.layers)):
            if self.use_layer_residual_format:
                zero_module(self.seqTransEncoder.layers[idx].norm2)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.use_pose_condition:
            self.embed_pose_cond = nn.Sequential(
                                    nn.Linear(self.num_feat, self.latent_dim),
                                    nn.GELU(),
                                    nn.Linear(self.latent_dim, self.latent_dim),
                                )

        if self.use_text_condition:
            self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            self.clip_version = clip_version
            self.load_and_freeze_clip(self.clip_version)

        self.output_process = OutputProcess(self.num_feat, self.latent_dim)

    def update_cond_dict(self, batch, cond_dict):
        if self.use_pose_condition or (not self.training and self.inpaint_gt_first_pose):
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

            # TODO: condition augmentation?
            # first pose in the sequence gives the condition
            cond_dict['pose_cond'] = torch.cat(
                [batch[idx, cond_idx[idx]:(cond_idx[idx] + 1)] for idx in range(batch.shape[0])], dim=0,
            )
        return cond_dict

    def denoised_fn(self, batch, cond_dict):
        if self.inpaint_gt_first_pose:
            if self.use_pose_condition_first_frame:
                # first frame condition
                cond_idx = torch.zeros([batch.shape[0]]).long().to(batch.device)
            else:
                # middle frame condition
                cond_idx = cond_dict['lengths'] // 2
            # overwrite model denoised predictions with ground truth first pose
            for idx in range(batch.shape[0]):
                batch[idx, cond_idx[idx]] = cond_dict['pose_cond']
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

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        # truncate beyond 77 tokens
        texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length]
        return self.clip_model.encode_text(texts).float()

    def _prepare_emb(self, timesteps, text=None, text_encoding=None, pose_cond=None):
        emb = self.embed_timestep(timesteps)  # [batch_size, model_feat]
        if self.use_text_condition:
            assert text is not None or text_encoding is not None
            if text_encoding is None:
                enc_text = self.encode_text(text) # [batch_size, 512]
            else:
                # precomputed encoding, for faster test-time sampling
                enc_text = text_encoding
            emb_text = self.embed_text(enc_text)  # [batch_size, model_feat]
            emb = emb + emb_text  # [batch_size, model_feat]
        if self.use_pose_condition:
            emb_pose_cond = self.embed_pose_cond(pose_cond)  # [batch_size, model_feat]
            emb = emb + emb_pose_cond
        emb = emb.unsqueeze(1)  # [batch_size, 1, model_feat]. treated as first token of sequence
        return emb

    def _apply_layer(self, x, mask, layer):
        if self.use_layer_residual_format:
            x_res = x
            if self.use_positional_encoding:
                x_res = self.sequence_pos_encoder(x_res)
            x_res = layer(x_res, src_key_padding_mask=mask)
            x = x + x_res
        else:
            x = layer(x, src_key_padding_mask=mask)
        return x

    def forward(self, x, timesteps, **cond_dict):
        """
        x: [batch_size, seq_len, pose_feat]
        timesteps: [batch_size] (int)
        cond_dict: dict with keys:
            'text': list of raw texts for each batch sample
            'lengths': [batch_size] or None, gives number of motion frames
            'pose_mask': [batch_size] or None, gives number of poses
        """
        assert ('lengths' in cond_dict.keys() or 'num_poses' in cond_dict.keys())
        assert not ('lengths' in cond_dict.keys() and 'num_poses' in cond_dict.keys()), \
            'Only one type of sequence allowed in cond dict.'
        seq_lens = (cond_dict['lengths'] if 'lengths' in cond_dict.keys() else cond_dict['num_poses'])

        # conditions and masks
        if self.training and self.cond_mask_prob > 0:
            cond_dict['text'] = apply_text_cond_mask(cond_dict['text'], self.cond_mask_prob)
        emb = self._prepare_emb(
            timesteps=timesteps,
            text=(cond_dict['text'] if 'text' in cond_dict.keys() else None),
            text_encoding=cond_dict['text_encoding'] if 'text_encoding' in cond_dict.keys() else None,
            pose_cond=(cond_dict['pose_cond'] if self.use_pose_condition else None),
        )
        mask = make_mask(seq_lens=seq_lens, max_len=x.shape[1], device=x.device)
        # record the max length of the input sample
        # note that this could decrease when going through the transformer depending on the masking for the batch
        max_len = x.shape[1]

        x = self.input_process(x)  # [batch_size, seq_len+1, model_feat]
        x = join_pose_and_cond(x, emb)  # [batch_size, seq_len+1, model_feat]
        if self.use_positional_encoding and not self.use_layer_residual_format:
            x = self.sequence_pos_encoder(x)  # [batch_size, seq_len+1 model_feat]

        # transformer body, x keeps same shape
        for idx in range(len(self.seqTransEncoder.layers)):
            x = self._apply_layer(x, mask=mask, layer=self.seqTransEncoder.layers[idx])

        # remove emb token, project to pose shape
        x, _ = separate_pose_and_cond(x)  # [batch_size, seq_len, model_feat]
        output = self.output_process(x)  # [batch_size, seq_len, pose_feat]
        if not self.training:
            # pad output in eval mode
            output = get_pose_pad_batch(output, lengths=seq_lens, max_len=max_len)
        return output


if __name__ == "__main__":

    from pose_datasets.data_utils import set_dummy_poses_to_zero

    joint_net = JointNet(dropout=0.0).cuda()

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
