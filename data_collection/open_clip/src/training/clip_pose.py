""" 
CLIP Model for Pose
"""
from dataclasses import dataclass
from collections import OrderedDict
import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

import sys
sys.path.append('../pose_datasets/')
sys.path.append('../../../pose_datasets/')
from clip_tokenizer import tokenize as clip_tokenizer


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False


DEFAULT_CONFIG = {
    "embed_dim": 768,
    "text_cfg": {
        "context_length": 77,
        "vocab_size": 49408,
        "width": 768,
        "heads": 12,
        "layers": 12,
    },
    "clpp_cfg": {
        "context_length": 10,
        "vocab_size": 10,  # dummy arg
        "width": 768,
        "heads": 12,
        "layers": 12,
    },
    "clmp_cfg": {
        "context_length": 61,
        "vocab_size": 10,  # dummy arg
        "width": 768,
        "heads": 4,
        "layers": 8,
    },
}


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ls_init_value: float = None,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            embed_cls: bool = False,
            pad_id: int = 0,
            output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        cast_dtype=None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=nn.GELU,
        norm_layer=norm_layer,
    )

    return text


def setup_clpp(
    device=None,
    weight_path="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/clip_pose/pose/2024_02_16-02_25_02-model_RN50-lr_0.0005-b_320-j_4-p_amp/checkpoints/epoch_1.pt",
):
    if device is None:
        device = 'cuda'
    encoder = CLPP(output_dict=False, clpp_config=True)
    sd = torch.load(weight_path, map_location="cpu")['state_dict']
    # remove ddp "module" string from beginning of state dict keys
    sd_clean = {}
    for key in sd.keys():
        key_clean = ".".join(key.split(".")[1:])
        sd_clean[key_clean] = sd[key]
    encoder.load_state_dict(sd_clean)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder


def setup_clmp(
    device=None,
    # weight path below is for motion samples with 61 frames
    weight_path="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/clip_pose/pose/2024_02_27-00_57_15-model_RN50-lr_0.0005-b_128-j_4-p_amp/checkpoints/epoch_19.pt",
    # weight path below is for motion samples with 120 frames
    #weight_path="/nfs/USRCSEA/IVA/Experiments/motion-diffusion/clip_pose/pose/2024_02_17-02_35_05-model_RN50-lr_0.0005-b_128-j_4-p_amp/checkpoints/epoch_17.pt",
):
    if weight_path is not None:
        sd = torch.load(weight_path, map_location="cpu")['state_dict']
        # get the number of frames used to train the model
        num_frames = sd['module.positional_embedding_pose'].shape[0]
        # remove ddp "module" string from beginning of state dict keys
        sd_clean = {}
        for key in sd.keys():
            key_clean = ".".join(key.split(".")[1:])
            sd_clean[key_clean] = sd[key]
    else:
        sd_clean = None
        num_frames = None

    encoder = CLPP(output_dict=False, clpp_config=False, num_frames=num_frames)
    if sd_clean is not None:
        encoder.load_state_dict(sd_clean)
    encoder = encoder.to(device if device is not None else 'cuda')
    encoder.eval()

    return encoder


# Constrastive Language Pose Model
class CLPP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        config_dict=None,
        cast_dtype=None,
        output_dict=True,
        pose_dim=158,
        clpp_config=True,
        num_frames=None,
        clmp_freeze_text_tower=False,
    ):
        super().__init__()
        if config_dict is None:
            config_dict = DEFAULT_CONFIG
        embed_dim = config_dict["embed_dim"]
        text_cfg = config_dict["text_cfg"]
        if clpp_config:
            pose_cfg = config_dict["clpp_cfg"]
        else:
            pose_cfg = config_dict["clmp_cfg"]
        embed_dim_pose = pose_cfg["width"]
        self.output_dict = output_dict

        text = _build_text_tower(embed_dim, text_cfg, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)
        if (not clpp_config) and clmp_freeze_text_tower:
            # TODO: there will be a bug if the line below is used outside of training
            from open_clip.factory import create_model_and_transforms
            open_clip_model, _, _ = create_model_and_transforms(
                'ViT-L-14',
                '/nfs/USRCSEA/IVA/Models/CLIP/OpenCLIP-Datacomp/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin',
                precision='amp',
                device='cpu',
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=None,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                aug_cfg={},
                output_dict=True,
            )

            self.transformer.load_state_dict(open_clip_model.transformer.state_dict())
            self._freeze_layer(self.transformer)

            self.token_embedding.load_state_dict(open_clip_model.token_embedding.state_dict())
            self._freeze_layer(self.token_embedding)

            self.ln_final.load_state_dict(open_clip_model.ln_final.state_dict())
            self._freeze_layer(self.ln_final)

            self.text_projection = open_clip_model.text_projection
            self.text_projection.requires_grad = False

        if num_frames is not None:
            pose_cfg['context_length'] = num_frames
        pose = _build_text_tower(embed_dim_pose, pose_cfg, cast_dtype)
        self.pose_linear_in = nn.Linear(pose_dim, embed_dim_pose)
        self.transformer_pose = pose.transformer
        self.context_length_pose = pose.context_length
        self.positional_embedding_pose = pose.positional_embedding
        self.ln_final_pose = pose.ln_final
        self.pose_projection = pose.text_projection
        #self.register_buffer('attn_mask_pose', pose.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def _freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def encode_pose(self, pose, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.pose_linear_in(pose).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding_pose.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_pose(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final_pose(x)  # [batch_size, n_ctx, transformer.width]

        # take features from first pose token (TODO: improve this?)
        x = x[torch.arange(x.shape[0]), 0] @ self.pose_projection
        return F.normalize(x, dim=-1) if normalize else x

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def zero_pad_to_context_length(self, pose):
        pad_size = self.context_length_pose - pose.shape[1]
        pose_pad = torch.zeros([pose.shape[0], pad_size, pose.shape[-1]]).to(pose.device)
        pose = torch.cat((pose, pose_pad), 1)
        return pose

    def encode_pose_and_text(
        self,
        pose,
        text,
        batch_size=None,
        device='cuda',
        normalize=True,
    ):
        if batch_size is None:
            batch_size = pose.shape[0]
        assert len(pose.shape) == 3  # [batch_size, num_poses, 158]
        assert pose.shape[-1] == 158

        if pose.shape[1] < self.context_length_pose:
            pose = self.zero_pad_to_context_length(pose)

        pose_features = []
        text_features = []
        for batch in range(int(np.ceil(pose.shape[0] / batch_size))):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, pose.shape[0])
            with torch.no_grad():
                feats_batch = self.encode_pose(pose[batch_start:batch_end].to(device), normalize=normalize)
            pose_features.append(feats_batch.cpu())
        for batch in range(int(np.ceil(len(text) / batch_size))):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, len(text))
            tokens = clip_tokenizer(text[batch_start:batch_end]).to(device)
            with torch.no_grad():
                feats_batch = self.encode_text(tokens, normalize=normalize)
            text_features.append(feats_batch.cpu())
        pose_features = torch.cat(pose_features, 0)
        text_features = torch.cat(text_features, 0)

        return pose_features, text_features

    def forward(
            self,
            pose: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        pose_features = self.encode_pose(pose, normalize=True) if pose is not None else None
        if self.output_dict:
            return {
                "image_features": pose_features,  # keep name "image_features" in dict for compatibility
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return pose_features, text_features, self.logit_scale.exp()


if __name__ == "__main__":

    # python3 clip_pose.py

    import sys
    sys.path.append('../../../../pose_datasets')
    from laion_pose_loader import LaionPose
    from data_utils import convert_6d_to_smpl, reverse_align_laion_pose, get_smpl_params_from_pose
    from clip_tokenizer import tokenize as clip_tokenizer

    pose_data = LaionPose(
                    batch_size=10,
                    resampled=True,
                    concat_vars=True,
                    yield_cond_dict=True,
                    use_dummy_betas=True,
                )
    data_iterator = iter(pose_data)

    encoder = setup_clpp(device='cuda')
    pose, cond = next(data_iterator)

    text = cond['text']
    pose_features, text_features = encoder.encode_pose_and_text(pose, text, device='cuda')
    sim_pairs = encoder.get_sim(pose_features, text_features)
    sim_all = (pose_features @ text_features.permute(1, 0))
    print(sim_pairs)
    print(sim_all)
