# original image-based unet, for reference.

from abc import abstractmethod

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: add relative positional encoding, mlp for embedding, switch for temporal attn

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


class SpatialReshape2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.height = None
        self.width = None
    def forward(self, x):
        in_shape = x.shape
        if len(in_shape) == 4:
            # set height and width on first fwd pass
            if self.height is None:
                self.height = in_shape[2]
            if self.width is None:
                self.width = in_shape[3]
            # reshape (B, C, H, W) to (B, C, H * W)
            x = x.reshape(in_shape[0], in_shape[1], -1).contiguous()
        elif len(in_shape) == 3:
            # reshape (B, C, H * W) to (B, C, H, W)
            x = x.reshape(in_shape[0:2] + (self.height, self.width)).contiguous()
        else:
            raise ValueError('Input tensor must be either 3D tensor or 4D tensor.')
        return x


class SpatialReshape3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.frames = None
        self.height = None
        self.width = None
    def forward(self, x):
        in_shape = x.shape
        if len(in_shape) == 5:
            # set frames, height and width on first fwd pass
            if self.frames is None:
                self.frames = in_shape[2]
            if self.height is None:
                self.height = in_shape[3]
            if self.width is None:
                self.width = in_shape[4]
            # reshape (B, C, F, H, W) to (B * F, C, H * W)
            x = x.permute(0, 2, 1, 3, 4).reshape(-1, in_shape[1], in_shape[3] * in_shape[4]).contiguous()
        elif len(in_shape) == 3:
            # reshape (B * F, C, H * W) to (B, C, F, H, W)
            bfchw_sz = (-1, self.frames, in_shape[1], self.height, self.width)
            x = x.reshape(bfchw_sz).permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError('Input tensor must be either 3D tensor or 5D tensor.')
        return x


class TemporalReshape(nn.Module):
    def __init__(self):
        super().__init__()
        self.height = None
        self.width = None
    def forward(self, x):
        in_shape = x.shape
        if len(in_shape) == 5:
            # set height and width on first fwd pass
            if self.height is None:
                self.height = in_shape[3]
            if self.width is None:
                self.width = in_shape[4]
            # reshape (B, C, F, H, W) to (B * H * W, C, F)
            x = x.permute(0, 3, 4, 1, 2).reshape(-1, in_shape[1], in_shape[2]).contiguous()
        elif len(in_shape) == 3:
            # reshape (B * H * W, C, F) to (B, C, F, H, W)
            x = x.reshape((-1, self.height, self.width) + in_shape[1:3]).permute(0, 3, 4, 1, 2).contiguous()
        else:
            raise ValueError('Input tensor must be either 3D tensor or 5D tensor.')
        return x


def conv_nd(conv_type, *args, **kwargs):
    # NOTE: args order is (in_channel, out_channel, kernel_size)
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    assert conv_type in ['1d', '2d', '3d_full', '3d_pseudo', '3d_spatial']
    if conv_type == '1d':
        return nn.Conv1d(*args, **kwargs)
    elif conv_type == '2d':
        return nn.Conv2d(*args, **kwargs)
    elif conv_type == '3d_full':
        return nn.Conv3d(*args, **kwargs)
    elif conv_type == '3d_pseudo':
        # spatial kernel and padding (plus optional stride passed through kwargs)
        spatial_kernel_size = (1, args[2], args[2])
        kwargs['padding'] = (
            (0, kwargs['padding'], kwargs['padding']) if 'padding' in kwargs.keys()
            else 0
        )
        # temporal kernel and padding (use stride=1 default)
        temporal_kernel_size = (args[2], 1, 1)
        temporal_padding = (
            (kwargs['padding'], 0, 0) if 'padding' in kwargs.keys()
            else 0
        )
        return nn.Sequential(
            nn.Conv3d(args[0], args[1], spatial_kernel_size, **kwargs),  # spatial conv
            nn.Conv3d(args[0], args[1], temporal_kernel_size, padding=temporal_padding),  # temporal conv
        )
    elif conv_type == '3d_spatial':
        kernel_size = (1, args[2], args[2])
        kwargs['padding'] = (
            (0, kwargs['padding'], kwargs['padding']) if 'padding' in kwargs.keys()
            else 0
        )
        return nn.Conv3d(args[0], args[1], kernel_size, **kwargs)
    raise ValueError(f"unsupported conv_type: {conv_type}")


def attn_nd(attn_type, channels, **kwargs):
    assert attn_type in ['2d', '3d']
    if attn_type == '2d':
        spatial_reshape = SpatialReshape2D()
        return nn.Sequential(
            spatial_reshape,
            AttentionBlock(channels, **kwargs),
            spatial_reshape,
        )
    elif attn_type == '3d':
        spatial_reshape = SpatialReshape3D()
        temporal_reshape = TemporalReshape()
        return UNetSequential(
            # space attn
            spatial_reshape,
            AttentionBlock(channels, **kwargs),
            spatial_reshape,
            # time attn
            temporal_reshape,
            AttentionBlock(channels, temporal_layer=True, **kwargs),
            temporal_reshape,
        )


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(conv_type, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    assert conv_type.startswith('1d') or conv_type.startswith('2d') or conv_type.starts_with('3d')
    if conv_type.startswith('1d'):
        return nn.AvgPool1d(**kwargs)
    elif conv_type.startswith('2d'):
        return nn.AvgPool2d(**kwargs)
    elif conv_type.startswith('3d'):
        return nn.AvgPool3d(**kwargs)
    raise ValueError(f"unsupported conv_type: {conv_type}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


# TODO: add masking to TimeStepBlock for masking pseudo3d conv when that is used?
class MaskBlock(nn.Module):
    """
    Any module where forward() takes mask as a second argument.
    """

    @abstractmethod
    def forward(self, x, use_image_mask):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class UNetSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings and/or mask to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, use_image_mask):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, MaskBlock):
                x = layer(x, use_image_mask)
            elif isinstance(layer, UNetSequential):
                x = layer(x, emb, use_image_mask)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param conv_type: determines if the signal is 1D, 2D, or 3D. If 3D, then
                      upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, conv_type, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.conv_type = conv_type
        if use_conv:
            self.conv = conv_nd(self.conv_type, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.conv_type.startswith('3d'):
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param conv_type: determines if the signal is 1D, 2D, or 3D. If 3D, then
                      downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, conv_type, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.conv_type = conv_type
        stride = 2 if not self.conv_type.startswith('3d') else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                self.conv_type, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(self.conv_type, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param conv_type: determines if the signal is 1D, 2D, or 3D. For 3D, determines conv method.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        conv_type='2d',
        up=False,
        down=False,
        emb_mlp_layers=0,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.conv_type = conv_type
        self.emb_mlp_layers = emb_mlp_layers

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(self.conv_type, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, self.conv_type)
            self.x_upd = Upsample(channels, False, self.conv_type)
        elif down:
            self.h_upd = Downsample(channels, False, self.conv_type)
            self.x_upd = Downsample(channels, False, self.conv_type)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        emb_layers_list = []
        for _ in range(self.emb_mlp_layers):
            emb_layers_list.extend([
                nn.SiLU(),
                linear(emb_channels, emb_channels),
            ])
        emb_layers_list.extend([
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        ])
        self.emb_layers = nn.Sequential(*emb_layers_list)
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(self.conv_type, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                self.conv_type, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(self.conv_type, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


# from https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
# rearrange output to (channel, length_q, length_k)
# TODO: different embedding for different heads?
class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position=128):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q).to(self.embeddings_table.device)
        range_vec_k = torch.arange(length_k).to(self.embeddings_table.device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        embeddings = self.embeddings_table[final_mat.long()]

        return embeddings.permute([2, 0, 1])


class AttentionBlock(MaskBlock):
    """
    An attention block that allows spatial or temporal positions to attend to each other.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        temporal_layer=False,
        mask_val=-1.0e20,
    ):
        super().__init__()
        self.channels = channels
        self.mask_val = mask_val
        self.temporal_layer = temporal_layer
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        assert self.channels % self.num_heads == 0
        self.norm = normalization(channels)
        self.qkv = conv_nd('1d', channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd('1d', channels, channels, 1))
        self.pos_emb_k =(
            RelativePosition(num_units=self.channels // self.num_heads) if self.temporal_layer
            else None
        )
        self.pos_emb_v = (
            RelativePosition(num_units=self.channels // self.num_heads) if self.temporal_layer
            else None
        )

    def _get_mask(self, use_image_mask, num_tokens, batch_size):
        # TODO: make this cleaner with fewer reshape?
        image_mask = self.mask_val * (1.0 - torch.eye(num_tokens))
        image_mask = image_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.num_heads, 1, 1)
        
        use_image_mask = use_image_mask.reshape(-1, 1, 1, 1, 1)
        use_image_mask = use_image_mask.repeat(1, batch_size // use_image_mask.shape[0], 1, 1, 1)
        use_image_mask = use_image_mask.reshape(-1, 1, 1, 1)

        mask = use_image_mask.float() * image_mask.to(use_image_mask.device)
        mask = mask.reshape(batch_size * self.num_heads, mask.shape[2], mask.shape[3])
        return mask

    def forward(self, x, use_image_mask=None):
        # TODO: adapt this to also work as x-attn by adding optional context input arg?
        qkv = self.qkv(self.norm(x))
        pos_emb_k = (
            self.pos_emb_k(x.shape[-1], x.shape[-1]) if self.pos_emb_k is not None
            else None
        )
        pos_emb_v = (
            self.pos_emb_v(x.shape[-1], x.shape[-1]) if self.pos_emb_v is not None
            else None
        )

        if use_image_mask is not None and self.temporal_layer:
            # mask attention between temporal tokens for joint video-image training
            mask = self._get_mask(use_image_mask, x.shape[-1], x.shape[0])
            mask = mask.type(x.dtype)
        else:
            mask = None

        h = self.attention.forward(qkv, mask, pos_emb_k, pos_emb_v)
        h = self.proj_out(h)
        return x + h


class QKVAttention():
    def __init__(self, n_heads):
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, pos_emb_k=None, pos_emb_v=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards

        if pos_emb_k is not None:
            weight_pos_k = torch.einsum(
                "bct,cts->bts",
                (q * scale).view(bs * self.n_heads, ch, length),
                (pos_emb_k * scale),
            )
            weight += weight_pos_k

        # mask
        if mask is not None:
            weight += mask

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))

        if pos_emb_v is not None:
            a_pos_v = torch.einsum("bts,cts->bct", weight, pos_emb_v)
            a += a_pos_v

        return a.reshape(bs, -1, length)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param conv_type: determines if the signal is 1D, 2D, or 3D. For 3D, determines conv method.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        image_size,
        frames,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        conv_type='2d',
        attn_type='2d',
        num_classes=None,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        emb_mlp_layers=0,
        use_scale_shift_norm=False,
        resblock_updown=False,
        final_zero_module=True,
    ):
        super().__init__()

        self.image_size = image_size
        self.frames = frames
        self.in_channels = in_channels
        if self.frames is None:
            self.state_dims = [self.in_channels, self.image_size, self.image_size]
        else:
            self.state_dims = [self.in_channels, self.frames, self.image_size, self.image_size]
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.emb_mlp_layers = emb_mlp_layers
        self.conv_type = conv_type
        self.attn_type = attn_type

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [UNetSequential(conv_nd(self.conv_type, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        conv_type=self.conv_type,
                        use_scale_shift_norm=use_scale_shift_norm,
                        emb_mlp_layers=self.emb_mlp_layers,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        attn_nd(
                            self.attn_type,
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                self.input_blocks.append(UNetSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    UNetSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            conv_type=self.conv_type,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            emb_mlp_layers=self.emb_mlp_layers,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, conv_type=self.conv_type, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = UNetSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                conv_type=self.conv_type,
                use_scale_shift_norm=use_scale_shift_norm,
                emb_mlp_layers=self.emb_mlp_layers,
            ),
            attn_nd(
                self.attn_type,
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                conv_type=self.conv_type,
                use_scale_shift_norm=use_scale_shift_norm,
                emb_mlp_layers=self.emb_mlp_layers,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        conv_type=self.conv_type,
                        use_scale_shift_norm=use_scale_shift_norm,
                        emb_mlp_layers=self.emb_mlp_layers,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        attn_nd(
                            self.attn_type,
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            conv_type=self.conv_type,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            emb_mlp_layers=self.emb_mlp_layers,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, conv_type=self.conv_type, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(UNetSequential(*layers))
                self._feature_size += ch

        final_conv = conv_nd(self.conv_type, input_ch, out_channels, 3, padding=1)
        if final_zero_module:
            final_conv = zero_module(final_conv)

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            final_conv,
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, use_image_mask=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, use_image_mask)
            hs.append(h)
        h = self.middle_block(h, emb, use_image_mask)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, use_image_mask)
        h = h.type(x.dtype)
        return self.out(h)


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = torch.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()
    if hasattr(l, 'embeddings_table') and l.embeddings_table is not None:
        l.embeddings_table.data = (
            l.embeddings_table.data.half()
        )


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()
    if hasattr(l, 'embeddings_table') and l.embeddings_table is not None:
        l.embeddings_table.data = (
            l.embeddings_table.data.float()
        )


if __name__ == "__main__":
    # test to check whether 3D unet is permutation-invariant across frames when use_image_mask input arg is True
    # NOTE: to make this work, need to remove zero_module from final layer, else all outputs are 0
    model = UNetModel(
        image_size=64,
        frames=16,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=3,
        attention_resolutions=(2, 4, 8),
        channel_mult=(1, 2, 4, 8),
        conv_type="3d_spatial",  # TODO: add ability for image mode with "3d_pseudo" conv_type
        attn_type="3d",
        num_classes=None,
        num_heads=4,
        final_zero_module=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
    )
    # need to experiment with trained model, otherwise diff is small by default for rand init net
    model.load_state_dict(
        torch.load(
            "/home/notebook/data/group/video-diffusion/tests-v0/ucf101/default_config_vid_32_gpu/23-06-30-20-43-42/checkpoints/model010000.pt"
        )
    )

    # first vid treated as independent frames, second vid has temporal connection between frames
    use_image_mask = torch.tensor([True, False])

    # dummy vids
    vid1a = torch.randn([1, 3, 16, 32, 32])
    vid2a = torch.randn([1, 3, 16, 32, 32])
    vid_a = torch.cat((vid1a, vid2a), 0)

    # shuffle frames of each vid
    rperm_1 = torch.randperm(vid1a.shape[2])
    rperm_2 = torch.randperm(vid2a.shape[2])
    vid1b = vid1a[:, :, rperm_1]
    vid2b = vid2a[:, :, rperm_2]
    vid_b = torch.cat((vid1b, vid2b), 0)

    # model predictions
    timesteps = torch.zeros([2]).long()
    out_a = model(vid_a, timesteps, use_image_mask=use_image_mask)
    out_b = model(vid_b, timesteps, use_image_mask=use_image_mask)

    # output for second set of inputs
    out_1b = out_b[0]
    out_2b = out_b[1]

    # shuffle outputs of original vids with same shuffle as second set of inputs
    out_1a_perm = out_a[0, :, rperm_1]
    out_2a_perm = out_a[1, :, rperm_2]

    print("output diff for image mode (should be 0 up to numerical error): ", (out_1a_perm - out_1b).abs().max())
    print("output diff for video mode (should be nonzero): ", (out_2a_perm - out_2b).abs().max())
