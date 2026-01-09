# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


import os
from typing import Optional, Union, List

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .attention_processors import CrossAttentionProcessor
from ...utils import logger
from torch_cluster import fps

from diffusers.models.embeddings import get_1d_rotary_pos_embed, apply_rotary_emb

import numpy as np

scaled_dot_product_attention = nn.functional.scaled_dot_product_attention

if os.environ.get('USE_SAGEATTN', '0') == '1':
    try:
        from sageattention import sageattn
    except ImportError:
        raise ImportError('Please install the package "sageattention" to use this USE_SAGEATTN.')
    scaled_dot_product_attention = sageattn


class FourierEmbedder(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Denote the space is [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, 3 / num_freqs, ..., (num_freqs - 1) / num_freqs].
    If logspace is True, then the frequency f_i is [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...];
    Otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)].

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:

        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class MLP(nn.Module):
    def __init__(
        self, *,
        width: int,
        expand_ratio: int = 4,
        output_width: int = None,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, output_width if output_width is not None else width)
        self.gelu = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        n_data: Optional[int] = None,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

        self.attn_processor = CrossAttentionProcessor()

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = self.attn_processor(self, q, k, v)
        out = out.transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        kv_cache: bool = False,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads,
            n_data=n_data,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.kv_cache = kv_cache
        self.data = None

    def forward(self, x, data):
        x = self.c_q(x)
        if self.kv_cache:
            if self.data is None:
                self.data = self.c_kv(data)
                logger.info('Save kv cache,this should be called only once for one mesh')
            data = self.data
        else:
            data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        data_width: Optional[int] = None,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = norm_layer(data_width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        n_ctx: int,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, qkv, rotary_emb=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        # import pdb;pdb.set_trace()

        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))

        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb)
            k = apply_rotary_emb(k, rotary_emb)

        out = scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(
            heads=heads,
            n_ctx=n_ctx,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, rotary_emb=None):
        x = self.c_qkv(x)
        x = self.attention(x, rotary_emb = rotary_emb)
        x = self.drop_path(self.c_proj(x))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
        use_temporal_attention: bool = False,
        use_global_attention: bool = False,
    ):
        super().__init__()
        self.use_temporal_attention = use_temporal_attention
        self.use_global_attention = use_global_attention
        self.heads = heads
        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)
        self.ln_2 = norm_layer(width, elementwise_affine=True, eps=1e-6)

        if use_temporal_attention:
            self.attn_t = MultiheadAttention(
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                qk_norm=qk_norm,
                drop_path_rate=drop_path_rate
            )
            self.attn_t.c_proj.weight.data.zero_()
            self.attn_t.c_proj.bias.data.zero_()
            self.ln_t = norm_layer(width, elementwise_affine=True, eps=1e-6)
        
        if use_global_attention:
            self.attn_g = MultiheadAttention(
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                qk_norm=qk_norm,
                drop_path_rate=drop_path_rate
            )
            self.attn_g.c_proj.weight.data.zero_()
            self.attn_g.c_proj.bias.data.zero_()
            self.ln_g = norm_layer(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor, rotary_emb=None, global_rotary_emb=None):
        assert len(x.shape) == 4
        B, T, N, C = x.shape
        if self.use_temporal_attention:
            x = x.permute(0,2,1,3).reshape(B*N, T, C)
            norm_x = self.ln_t(x)
            # import pdb;pdb.set_trace()
            # temporal_rope_emb = get_1d_rotary_pos_embed(int(C/self.heads), np.arange(T, dtype=np.float32), use_real=True) # here 8 is num_attention_heads
            attn_output = self.attn_t(
                norm_x,
                rotary_emb=rotary_emb,
            )
            x = x + attn_output
            x = x.reshape(B, N, T, C).permute(0,2,1,3)
        
        if self.use_global_attention:
            x = x.reshape(B, T*N, C)
            norm_x = self.ln_g(x)

            # T_grid = torch.arange(T, device=x.device).reshape(T, 1)
            # T_grid = T_grid.repeat(1, N) # [T, N]
            # T_grid = T_grid.cpu().numpy()
            # global_rope_emb = get_1d_rotary_pos_embed(int(C/self.heads), T_grid.reshape(-1), use_real=True)
            assert global_rotary_emb is not None

            attn_output = self.attn_g(
                norm_x,
                rotary_emb=global_rotary_emb,
            )
            x = x + attn_output
            x = x.reshape(B, T, N, C)
        
        x = x.reshape(B*T, N, C)

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        x = x.reshape(B, T, N, C)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
        use_temporal_attention: bool = False,
        use_global_attention: bool = False,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate,
                    use_temporal_attention=use_temporal_attention,
                    use_global_attention=use_global_attention
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, **kwargs):
        for block in self.resblocks:
            x = block(x, **kwargs)
        return x


class CrossAttentionDecoder(nn.Module):

    def __init__(
        self,
        *,
        num_latents: int,
        out_channels: int,
        fourier_embedder: FourierEmbedder,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        downsample_ratio: int = 1,
        enable_ln_post: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary"
    ):
        super().__init__()

        self.enable_ln_post = enable_ln_post
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        if self.downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)
        if self.enable_ln_post == False:
            qk_norm = False
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            mlp_expand_ratio=mlp_expand_ratio,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm
        )

        if self.enable_ln_post:
            self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)
        self.label_type = label_type
        self.count = 0

    def set_cross_attention_processor(self, processor):
        self.cross_attn_decoder.attn.attention.attn_processor = processor

    def set_default_cross_attention_processor(self):
        self.cross_attn_decoder.attn.attention.attn_processor = CrossAttentionProcessor

    def forward(self, queries=None, query_embeddings=None, latents=None):
        if query_embeddings is None:
            query_embeddings = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        self.count += query_embeddings.shape[1]
        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)
        x = self.cross_attn_decoder(query_embeddings, latents)
        if self.enable_ln_post:
            x = self.ln_post(x)
        occ = self.output_proj(x)
        return occ


def fps_old(
    src: torch.Tensor,
    batch: Optional[Tensor] = None,
    ratio: Optional[Union[Tensor, float]] = None,
    random_start: bool = True,
    batch_size: Optional[int] = None,
    ptr: Optional[Union[Tensor, List[int]]] = None,
):
    src = src.float()
    from torch_cluster import fps as fps_fn
    output = fps_fn(src, batch, ratio, random_start, batch_size, ptr)
    return output


class PointCrossAttentionEncoder(nn.Module):

    def __init__(
        self, *,
        num_latents: int,
        downsample_ratio: float,
        pc_size: int,
        pc_sharpedge_size: int,
        fourier_embedder: FourierEmbedder,
        point_feats: int,
        width: int,
        heads: int,
        layers: int,
        normal_pe: bool = False,
        qkv_bias: bool = True,
        use_ln_post: bool = False,
        use_checkpoint: bool = False,
        qk_norm: bool = False
    ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.downsample_ratio = downsample_ratio
        self.point_feats = point_feats
        self.normal_pe = normal_pe

        if pc_sharpedge_size == 0:
            print(
                f'PointCrossAttentionEncoder INFO: pc_sharpedge_size is zero')
        else:
            print(
                f'PointCrossAttentionEncoder INFO: pc_sharpedge_size is given, using pc_size={pc_size}, pc_sharpedge_size={pc_sharpedge_size}')

        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size

        self.fourier_embedder = fourier_embedder

        self.input_proj = nn.Linear((self.fourier_embedder.out_dim + point_feats) * 2, width)

        self.bone_linear = nn.Linear(4*self.fourier_embedder.out_dim, (self.fourier_embedder.out_dim + point_feats) * 2, bias=True)
        # zero init bone_linear weight
        nn.init.zeros_(self.bone_linear.weight)
        nn.init.zeros_(self.bone_linear.bias)

        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm
        )

        self.self_attn = None
        if layers > 0:
            self.self_attn = Transformer(
                n_ctx=num_latents,
                width=width,
                layers=layers,
                heads=heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                use_temporal_attention=True,
                use_global_attention=True
            )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None

    def sample_points_and_latents(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        B, N, D = pc.shape
        num_pts = self.num_latents * self.downsample_ratio

        # Compute number of latents
        num_latents = int(num_pts / self.downsample_ratio)

        # Compute the number of random and sharpedge latents
        num_random_query = self.pc_size / (self.pc_size + self.pc_sharpedge_size) * num_latents
        num_sharpedge_query = num_latents - num_random_query

        # Split random and sharpedge surface points
        random_pc, sharpedge_pc = torch.split(pc, [self.pc_size, self.pc_sharpedge_size], dim=1)
        assert random_pc.shape[1] <= self.pc_size, "Random surface points size must be less than or equal to pc_size"
        assert sharpedge_pc.shape[
                   1] <= self.pc_sharpedge_size, "Sharpedge surface points size must be less than or equal to pc_sharpedge_size"

        # Randomly select random surface points and random query points
        input_random_pc_size = int(num_random_query * self.downsample_ratio)
        random_query_ratio = num_random_query / input_random_pc_size
        idx_random_pc = torch.randperm(random_pc.shape[1], device=random_pc.device)[:input_random_pc_size]
        input_random_pc = random_pc[:, idx_random_pc, :]
        flatten_input_random_pc = input_random_pc.view(B * input_random_pc_size, D)
        N_down = int(flatten_input_random_pc.shape[0] / B)
        batch_down = torch.arange(B).to(pc.device)
        batch_down = torch.repeat_interleave(batch_down, N_down)
        idx_query_random = fps_old(flatten_input_random_pc, batch_down, ratio=random_query_ratio)
        query_random_pc = flatten_input_random_pc[idx_query_random].view(B, -1, D)

        # Randomly select sharpedge surface points and sharpedge query points
        input_sharpedge_pc_size = int(num_sharpedge_query * self.downsample_ratio)
        if input_sharpedge_pc_size == 0:
            input_sharpedge_pc = torch.zeros(B, 0, D, dtype=input_random_pc.dtype).to(pc.device)
            query_sharpedge_pc = torch.zeros(B, 0, D, dtype=query_random_pc.dtype).to(pc.device)
        else:
            sharpedge_query_ratio = num_sharpedge_query / input_sharpedge_pc_size
            idx_sharpedge_pc = torch.randperm(sharpedge_pc.shape[1], device=sharpedge_pc.device)[
                               :input_sharpedge_pc_size]
            input_sharpedge_pc = sharpedge_pc[:, idx_sharpedge_pc, :]
            flatten_input_sharpedge_surface_points = input_sharpedge_pc.view(B * input_sharpedge_pc_size, D)
            N_down = int(flatten_input_sharpedge_surface_points.shape[0] / B)
            batch_down = torch.arange(B).to(pc.device)
            batch_down = torch.repeat_interleave(batch_down, N_down)
            idx_query_sharpedge = fps_old(flatten_input_sharpedge_surface_points, batch_down, ratio=sharpedge_query_ratio)
            query_sharpedge_pc = flatten_input_sharpedge_surface_points[idx_query_sharpedge].view(B, -1, D)

        # Concatenate random and sharpedge surface points and query points
        query_pc = torch.cat([query_random_pc, query_sharpedge_pc], dim=1)
        input_pc = torch.cat([input_random_pc, input_sharpedge_pc], dim=1)

        # PE
        query = self.fourier_embedder(query_pc)
        data = self.fourier_embedder(input_pc)

        # Concat normal if given
        if self.point_feats != 0:

            random_surface_feats, sharpedge_surface_feats = torch.split(feats, [self.pc_size, self.pc_sharpedge_size],
                                                                        dim=1)
            input_random_surface_feats = random_surface_feats[:, idx_random_pc, :]
            flatten_input_random_surface_feats = input_random_surface_feats.view(B * input_random_pc_size, -1)
            query_random_feats = flatten_input_random_surface_feats[idx_query_random].view(B, -1,
                                                                                           flatten_input_random_surface_feats.shape[
                                                                                               -1])

            if input_sharpedge_pc_size == 0:
                input_sharpedge_surface_feats = torch.zeros(B, 0, self.point_feats,
                                                            dtype=input_random_surface_feats.dtype).to(pc.device)
                query_sharpedge_feats = torch.zeros(B, 0, self.point_feats, dtype=query_random_feats.dtype).to(
                    pc.device)
            else:
                input_sharpedge_surface_feats = sharpedge_surface_feats[:, idx_sharpedge_pc, :]
                flatten_input_sharpedge_surface_feats = input_sharpedge_surface_feats.view(B * input_sharpedge_pc_size,
                                                                                           -1)
                query_sharpedge_feats = flatten_input_sharpedge_surface_feats[idx_query_sharpedge].view(B, -1,
                                                                                                        flatten_input_sharpedge_surface_feats.shape[
                                                                                                            -1])

            query_feats = torch.cat([query_random_feats, query_sharpedge_feats], dim=1)
            input_feats = torch.cat([input_random_surface_feats, input_sharpedge_surface_feats], dim=1)

            if self.normal_pe:
                query_normal_pe = self.fourier_embedder(query_feats[..., :3])
                input_normal_pe = self.fourier_embedder(input_feats[..., :3])
                query_feats = torch.cat([query_normal_pe, query_feats[..., 3:]], dim=-1)
                input_feats = torch.cat([input_normal_pe, input_feats[..., 3:]], dim=-1)

            query = torch.cat([query, query_feats], dim=-1)
            data = torch.cat([data, input_feats], dim=-1)

        if input_sharpedge_pc_size == 0:
            query_sharpedge_pc = torch.zeros(B, 1, D).to(pc.device)
            input_sharpedge_pc = torch.zeros(B, 1, D).to(pc.device)

        # print(f'query_pc: {query_pc.shape}')
        # print(f'input_pc: {input_pc.shape}')
        # print(f'query_random_pc: {query_random_pc.shape}')
        # print(f'input_random_pc: {input_random_pc.shape}')
        # print(f'query_sharpedge_pc: {query_sharpedge_pc.shape}')
        # print(f'input_sharpedge_pc: {input_sharpedge_pc.shape}')

        return query.view(B, -1, query.shape[-1]), data.view(B, -1, data.shape[-1]), [query_pc, input_pc,
                                                                                      query_random_pc, input_random_pc,
                                                                                      query_sharpedge_pc,
                                                                                      input_sharpedge_pc]


    def _sample_features(
        self, x: torch.Tensor, num_tokens: int = 2048, seed: Optional[int] = None
    ):
        """
        Sample points from features of the input point cloud.

        Args:
            x (torch.Tensor): The input point cloud. shape: (B, N, C)
            num_tokens (int, optional): The number of points to sample. Defaults to 2048.
            seed (Optional[int], optional): The random seed. Defaults to None.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            x.shape[1], num_tokens * 4, replace=num_tokens * 4 > x.shape[1]
        )
        selected_points = x[:, indices]

        batch_size, num_points, num_channels = selected_points.shape
        flattened_points = selected_points.view(batch_size * num_points, num_channels)
        batch_indices = (
            torch.arange(batch_size).to(x.device).repeat_interleave(num_points)
        )

        # fps sampling
        sampling_ratio = 1.0 / 4
        sampled_indices = fps(
            flattened_points[:, :3],
            batch_indices,
            ratio=sampling_ratio,
            random_start=self.training,
        )
        sampled_points = flattened_points[sampled_indices].view(
            batch_size, -1, num_channels
        )

        return sampled_points

    def _sample_features_index(
        self, x: torch.Tensor, num_tokens: int = 2048, seed: Optional[int] = None
    ):
        """
        Sample points from features of the input point cloud.

        Args:
            x (torch.Tensor): The input point cloud. shape: (B, N, C)
            num_tokens (int, optional): The number of points to sample. Defaults to 2048.
            seed (Optional[int], optional): The random seed. Defaults to None.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            x.shape[1], num_tokens * 4, replace=num_tokens * 4 > x.shape[1]
        )
        selected_points = x[:, indices]

        batch_size, num_points, num_channels = selected_points.shape
        flattened_points = selected_points.view(batch_size * num_points, num_channels)
        batch_indices = (
            torch.arange(batch_size).to(x.device).repeat_interleave(num_points)
        )

        # fps sampling
        sampling_ratio = 1.0 / 4
        sampled_indices = fps(
            flattened_points[:, :3],
            batch_indices,
            ratio=sampling_ratio,
            random_start=self.training,
        )
        sampled_points = flattened_points[sampled_indices].view(
            batch_size, -1, num_channels
        )

        return indices, sampled_indices


    def sample_points_and_latents_triposg(self, x: torch.Tensor, num_tokens: int = 256, seed: Optional[int] = None, bone_positions_sequence_all=None):
        assert len(x.shape) == 4
        B, T, N, C = x.shape
        x = x.reshape(B*T, N, C)
        positions_1, features_1, positions_2, features_2 = x[..., :3], x[..., 3:6], x[..., 6:9], x[..., 9:12]
        # shape_input = torch.cat([batch['surface_src'][:,0], torch.zeros_like(batch['surface_src'][:,0,:,-1:])], dim=-1)
        x_kv = torch.cat([self.fourier_embedder(positions_1), features_1, torch.zeros_like(features_1[...,-1:]), self.fourier_embedder(positions_2), features_2, torch.zeros_like(features_2[...,-1:])], dim=-1)
        x = x.reshape(B, T, N, C)

        # import pdb;pdb.set_trace()

        if num_tokens < N:
            x32 = x[:,0].to(torch.float32)
            indices, sampled_indices = self._sample_features_index(x32, num_tokens, seed)
            x = x.reshape(B*T, -1, C)
            selected_points = x[:, indices]
            selected_points = selected_points.reshape(B, T, -1, C)
            selected_points = selected_points.permute(0, 2, 1, 3) # B, N, T, C
            selected_points = selected_points.reshape(-1, T, C)
            sampled_x = selected_points[sampled_indices]
            sampled_x = sampled_x.reshape(B, -1, T, C)
            sampled_x = sampled_x.permute(0, 2, 1, 3) # B, T, N, C
            sampled_x = sampled_x.reshape(B*T, -1, C)
            positions_1_s, features_1_s, positions_2_s, features_2_s = sampled_x[..., :3], sampled_x[..., 3:6], sampled_x[..., 6:9], sampled_x[..., 9:12]
            x_q = torch.cat([self.fourier_embedder(positions_1_s), features_1_s, torch.zeros_like(features_1_s[...,-1:]), self.fourier_embedder(positions_2_s), features_2_s, torch.zeros_like(features_2_s[...,-1:])], dim=-1)
            C = x_q.shape[-1]
            canonical_points = positions_1_s.reshape(B, T, -1, 3)[:,0]
        else:
            x = x.reshape(B*T, -1, C)
            positions_1, features_1, positions_2, features_2 = x[..., :3], x[..., 3:6], x[..., 6:9], x[..., 9:12]
            x_q = torch.cat([self.fourier_embedder(positions_1), features_1, torch.zeros_like(features_1[...,-1:]), self.fourier_embedder(positions_2), features_2, torch.zeros_like(features_2[...,-1:])], dim=-1)
            C = x_q.shape[-1]
            canonical_points = positions_1.reshape(B, T, -1, 3)[:,0]

        if bone_positions_sequence_all is not None:
            src_head, src_tail, tgt_head, tgt_tail = bone_positions_sequence_all[..., :3], bone_positions_sequence_all[..., 3:6], bone_positions_sequence_all[..., 6:9], bone_positions_sequence_all[..., 9:12]
            x_bone_q = torch.cat([self.fourier_embedder(src_head), self.fourier_embedder(src_tail), self.fourier_embedder(tgt_head), self.fourier_embedder(tgt_tail)], dim=-1)
            x_bone_q = self.bone_linear(x_bone_q)


        x_q = x_q.reshape(B, T, -1, C)
        if bone_positions_sequence_all is not None:
            x_q = torch.cat([x_q, x_bone_q], dim=-2)
        x_kv = x_kv.reshape(B, T, -1, C)
        # x = self.encoder(x_q, x_kv)
        # x = self.quant(x)
        return x_q, x_kv, canonical_points


    def forward(self, pc, num_tokens: int = 256, seed: Optional[int] = None, bone_positions_sequence_all=None, **kwargs):
        """

        Args:
            pc (torch.FloatTensor): [B, T, N, 3]

        Returns:

        """

        # query, data, pc_infos = self.sample_points_and_latents(pc, feats)
        query, data, canonical_points = self.sample_points_and_latents_triposg(pc, num_tokens, seed, bone_positions_sequence_all)


        query = self.input_proj(query)
        query = query
        data = self.input_proj(data)
        data = data

        B, T, N, C = query.shape

        query = query.reshape(B*T, -1, C)
        data = data.reshape(B*T, -1, C)

        latents = self.cross_attn(query, data)

        latents = latents.reshape(B, T, N, C)

        if self.self_attn is not None:
            latents = self.self_attn(latents, **kwargs)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, canonical_points
