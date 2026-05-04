import numpy as np

import torch
import torch.nn as nn

from pose_datasets.data_utils import get_masks_from_size


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def disable_layer_grad(layer):
    for param in layer.parameters():
        param.requires_grad = False


def enable_layer_grad(layer):
    for param in layer.parameters():
        param.requires_grad = True


def make_mask(seq_lens, max_len, device, reps=1):
    batch_size = len(seq_lens)
    # mask size: [batch_size, seq_len]
    mask = get_masks_from_size(seq_lens=seq_lens, max_len=max_len, device=device)
    text_mask = torch.zeros((batch_size, 1)).to(mask.device)
    # [batch_size, seq_len + 1]
    mask = torch.cat((text_mask, mask), dim=1).bool()
    if reps > 1:
        # repeat over pose dims to get shape [batch_size * reps, seq_len + 1]
        mask = mask.unsqueeze(1).repeat(1, reps, 1)
        mask = mask.view(batch_size * reps, -1).contiguous()
    return mask


def full_to_pose(x):
    x = x.view(-1, x.shape[2], x.shape[3])  # frames on batch dim
    return x


def pose_to_full(x, batch_size):
    x = x.view(batch_size, -1, x.shape[1], x.shape[2])  # frames back to separate dim
    return x


def full_to_motion(x):
    _, num_frames, _, num_feats = x.shape
    x = x.permute(0, 2, 1, 3).contiguous()  # move poses next to batch dim
    x = x.view(-1, num_frames, num_feats)  # poses on batch dim
    return x


def motion_to_full(x, batch_size):
    _, num_frames, num_feats = x.shape
    x = x.view(batch_size, -1, num_frames, num_feats)  # poses back to separate dim
    x = x.permute(0, 2, 1, 3).contiguous()  # poses back to original position
    return x


def split_by_seq_lens(tensor, seq_lens):
    motion_idx = (seq_lens > 1)
    pose_idx = (seq_lens == 1)
    assert motion_idx.sum() + pose_idx.sum() == tensor.shape[0]

    tensor_motion = tensor[motion_idx]
    tensor_pose = tensor[pose_idx]

    return tensor_motion, tensor_pose


def split_conds(cond_tensor, seq_lens):
    cond_in_shape = list(cond_tensor.shape)
    batch_size = len(seq_lens)
    assert cond_in_shape[0] % batch_size == 0
    cond_tensor = cond_tensor.view([batch_size, cond_in_shape[0] // batch_size] + cond_in_shape[1:])
    cond_motion, cond_pose = split_by_seq_lens(cond_tensor, seq_lens)
    cond_motion = cond_motion.view([-1] + cond_in_shape[1:])
    cond_pose = cond_pose.view([-1] + cond_in_shape[1:])
    return cond_motion, cond_pose


def split_motion_and_pose_data(x, emb_mo, emb_po, motion_mask, pose_mask, seq_lens):

    x_motion, x_pose = split_by_seq_lens(x, seq_lens)
    if x_motion.nelement() == 0:
        x_motion = None
    if x_pose.nelement() == 0:
        x_pose = None

    emb_mo_motion, emb_mo_pose = split_conds(emb_mo, seq_lens)
    emb_po_motion, emb_po_pose = split_conds(emb_po, seq_lens)
    motion_mask_motion, motion_mask_pose = split_conds(motion_mask, seq_lens)
    pose_mask_motion, pose_mask_pose = split_conds(pose_mask, seq_lens)

    # remove dummy sequence elements for pose data to increase efficiency
    if x_pose is not None:
        x_pose = x_pose[:, 0:1]
        emb_po_pose = emb_po_pose.view(x_pose.shape[0], x.shape[1], 1, x.shape[3])[:, 0]
        motion_mask_pose = motion_mask_pose[:, 0:2]
        pose_mask_pose = pose_mask_pose.view(x_pose.shape[0], x.shape[1], -1)[:, 0]

    return (
        x_motion, emb_mo_motion, emb_po_motion, motion_mask_motion, pose_mask_motion,
        x_pose, emb_mo_pose, emb_po_pose, motion_mask_pose, pose_mask_pose,
    )


def combine_tensors(tensor_list, id_list):
    # combine hidden states from different modeling streams
    # TODO: make this more efficient?
    if tensor_list[0] is None:
        return tensor_list[1]
    elif tensor_list[1] is None:
        return tensor_list[0]
    else:
        hidden_states_list = []
        idx_list = [0 for _ in tensor_list]
        for id_item in id_list:
            tensor = tensor_list[id_item.item()]
            idx = idx_list[id_item.item()]
            hidden_states_list.append(tensor[idx:(idx + 1)])
            idx_list[id_item.item()] = idx + 1
        for tensor, idx in zip(tensor_list, idx_list):
            # check that all tensors have been gathered
            assert tensor.shape[0] == idx
        hidden_states = torch.cat(hidden_states_list)
        return hidden_states


def join_pose_and_cond(x, emb):
    return torch.cat((emb, x), dim=1)


def separate_pose_and_cond(xseq):
    x = xseq[:, 1:]
    emb = xseq[:, :1]
    return x, emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, pad_pos_emb_start=False, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.pad_pos_emb_start = pad_pos_emb_start

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x, seq_lens=None):
        # x: [batch_size, frames, feat] or [batch_size * poses, frames, feat]
        # pe: [5000, feat]
        if self.training and seq_lens is not None and self.pad_pos_emb_start:

            # TODO: make more efficient?
 
            # randomly pad beginning of sequence according to sequence length
            tseq_starts = torch.cat(
                [
                    torch.randint(size=[1], high=max(1, x.shape[1] - int(seq_lens[idx])))
                    for idx in range(len(seq_lens))
                ],
                dim=0,
            ).to(x.device)
            # repeat across poses
            tseq_starts = tseq_starts.unsqueeze(1).repeat(1, x.shape[0] // tseq_starts.shape[0]).view(-1)

            pe = []
            for idx in range(x.shape[0]):
                tseq_start = int(tseq_starts[idx] + 1) # add one because first token is always for emb
                tseq_end = tseq_start + (x.shape[1] - 1)
                pe_emb = self.pe[0:1]  # first token for embedding
                pe_seq = self.pe[tseq_start:tseq_end]  # start seq pos emb after random pad
                pe.append(torch.cat((pe_emb, pe_seq), 0))
            pe = torch.stack(pe, 0) # [batch_size * poses, frames, feat]

        else:
            # pad always comes after visible sequence, add same pe to all elements of batch
            pe = self.pe[:x.shape[1]].unsqueeze(0)  # [1, frames, feat]

        x = x + pe
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.poseEmbedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x):
        x = self.poseEmbedding(x)
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.poseFinal = nn.Linear(latent_dim, input_feats)

    def forward(self, output):
        output = self.poseFinal(output)
        return output
