import torch
import torch.nn as nn
from typing import Optional
from einops import repeat
import math
from ThirdParty.michelangelo.models.modules.transformer_blocks import ResidualCrossAttentionBlock,Transformer, checkpoint
from torch.nn import Sequential, Dropout, Linear, ReLU, Parameter, BatchNorm1d
from typing import List, Optional, Tuple, Union

class ShapeAsLatentModule(nn.Module):
    latent_shape: Tuple[int, int]

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def query_geometry(self, *args, **kwargs):
        raise NotImplementedError
    
class FourierEmbedder(nn.Module):

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

        if self.num_freqs > 0:
            self.frequencies = self.frequencies.to(x.device)
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)

            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x
        
def MLP(channels, batch_norm=True):
    if batch_norm:
        return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU(), BatchNorm1d(channels[i], momentum=0.1))
                            for i in range(1, len(channels))])
    else:
        return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU()) for i in range(1, len(channels))])
    
class CrossAttentionEncoder(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 fourier_embedder: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        self.fourier_embedder = fourier_embedder
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width, device=device, dtype=dtype)
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        self.self_attn = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=False
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def _forward(self, pc, feats):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """

        bs = pc.shape[0]

        data = self.fourier_embedder(pc)
        if feats is not None:
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)

        query = repeat(self.query, "m c -> b m c", b=bs)
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            dict
        """

        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)
    
    
    
class TransformerEncoder(ShapeAsLatentModule):
    def __init__(self, *,
                 device: Optional[torch.device]='cuda',
                 dtype: Optional[torch.dtype],
                 num_latents: int = 16,
                 point_feats: int = 3,
                 embed_dim: int = 64,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int = 768,
                 heads: int = 12,
                 num_encoder_layers: int = 8,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False,
                 out_channels: int = 4):

        super().__init__()
        
        self.use_checkpoint = use_checkpoint

        self.num_latents = num_latents
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = CrossAttentionEncoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint
        )
        self.width = width
        self.out_channels = out_channels
        self.device = device
        
        self.embed_dim = embed_dim

    def encode(self,data):
        input_points = data['points_cloud'].to(self.device)
        bs = input_points.shape[0]
        pc, feats = input_points[...,:3], input_points[..., 3:]
        latents, _ = self.encoder(pc, feats)
        # print_time('after encoder')
        latents = latents.reshape(bs,-1, self.width)
        return latents
    def encode_pc(self,points_cloud):
        bs = points_cloud.shape[0]
        input_points = points_cloud.to(self.device)
        pc, feats = input_points[...,:3], input_points[..., 3:]
        latents, _ = self.encoder(pc, feats)
        
        latents = latents.reshape(bs,-1, self.width)
        return latents
    def forward(self, data):

        # input_points = torch.from_numpy(np.array(data.points_cloud)).cuda()
        input_points = data['points_cloud'].to(self.device)
        pc, feats = input_points[...,:3], input_points[..., 3:]
        latents, _ = self.encoder(pc, feats)
        
        latents = latents.reshape(-1, self.width)
        latents =latents.reshape(-1, self.num_latents, self.out_channels)
        latents[..., :3] = torch.tanh(latents[..., :3])
        latents[..., 3:] = torch.sigmoid(latents[..., 3:])
        
        
        return latents