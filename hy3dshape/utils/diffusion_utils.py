
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.utils import make_grid
import torch
from typing import List, Optional, Tuple, Union
import torch.nn as nn
import math
from timm.models.vision_transformer import Mlp, DropPath

def my_collate_diff(batch,return_joints_num=128,random=False):
    data = {}
    for key in batch[0]:
        if key=='vox' or key=='name' or key=='joints_num' or key=='skins_index' or key=='skins_weight' or key=='parent_index' or key=='conns' or key=='joints' or key=='bones' or key=='mesh_skins_index' or key=='mesh_skins_weight' or key=='mesh_pc' or key=='mesh_face':
            data[key] = [sample[key] for sample in batch]
        elif key=='pc':
            data['points_cloud'] = torch.stack([sample['pc'] for sample in batch])
        elif key=='skins':
            continue
        elif key=='bones_num':
            data[key] = torch.tensor([sample['bones_num'] for sample in batch])
        else:
            data[key] = torch.stack([sample[key] for sample in batch])

    if 'joints' in batch[0]:
        padded_joints_matrix = torch.ones(len(data['name']), return_joints_num, 3) * (-3)
        joints_matrix = torch.ones(len(data['name']), 96, 3) * (-3)
        for i in range(len(data['name'])):
            joints_matrix[i, :data['joints_num'][i], :] = data['joints'][i]
        if not random:
            for i in range(len(data['name'])):
                padded_joints_matrix[i] = data['joints'][i].repeat(return_joints_num//data['joints_num'][i]+1,1)[:return_joints_num,:] 
        else:
            for i in range(len(data['name'])):
                padded_joints_matrix[i] = data['joints'][i][torch.randint(0, data['joints_num'][i], (return_joints_num,))]
        data['joints_repeat'] = padded_joints_matrix
        data['joints'] = joints_matrix

    return data

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

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
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            kv_dim=None,
            num_heads=16,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        kv_dim = dim if not kv_dim else kv_dim
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B, N_q, C = x_q.shape
        B, N_kv, _ = x_kv.shape
        # [B, N_q, C] -> [B, N_q, H, C/H] -> [B, H, N_q, C/H]
        q = self.wq(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # [B, N_kv, C] -> [B, N_kv, H, C/H] -> [B, H, N_kv, C/H]
        k = self.wk(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # [B, N_kv, C] -> [B, N_kv, H, C/H] -> [B, H, N_kv, C/H]
        v = self.wv(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # [B, H, N_q, C/H] @ [B, H, C/H, N_kv] -> [B, H, N_q, N_kv]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # [B, H, N_q, N_kv] @ [B, H, N_kv, C/H] -> [B, H, N_q, C/H]
        x = attn @ v

        # [B, H, N_q, C/H] -> [B, N_q, C]
        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Compute_Block(nn.Module):

    def __init__(self, z_dim, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_z1 = norm_layer(z_dim)
        self.attn = CrossAttention(
            z_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_z2 = norm_layer(z_dim)
        mlp_hidden_dim = int(z_dim * mlp_ratio)
        self.mlp = Mlp(in_features=z_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, z):
        zn = self.norm_z1(z)
        z = z + self.drop_path(self.attn(zn, zn))
        z = z + self.drop_path(self.mlp(self.norm_z2(z)))
        return z

class Read_Block(nn.Module):

    def __init__(self, z_dim, x_dim, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_x = norm_layer(x_dim)
        self.norm_z1 = norm_layer(z_dim)
        self.attn = CrossAttention(
            z_dim, x_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_z2 = norm_layer(z_dim)
        mlp_hidden_dim = int(z_dim * mlp_ratio)
        self.mlp = Mlp(in_features=z_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, z, x):
        z = z + self.drop_path(self.attn(self.norm_z1(z), self.norm_x(x)))
        z = z + self.drop_path(self.mlp(self.norm_z2(z)))
        return z

class Write_Block(nn.Module):

    def __init__(self, z_dim, x_dim, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_z = norm_layer(z_dim)
        self.norm_x1 = norm_layer(x_dim)
        self.attn = CrossAttention(
            x_dim, z_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_x2 = norm_layer(x_dim)
        mlp_hidden_dim = int(x_dim * mlp_ratio)
        self.mlp = Mlp(in_features=x_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, z, x):
        x = x + self.drop_path(self.attn(self.norm_x1(x), self.norm_z(z)))
        x = x + self.drop_path(self.mlp(self.norm_x2(x)))
        return x

class RCW_Block(nn.Module):

    def __init__(self, z_dim, x_dim, num_compute_layers=4, num_heads=16, 
                 mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.read = Read_Block(z_dim, x_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, 
                                   attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
        self.write = Write_Block(z_dim, x_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, 
                                   attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
        self.compute = nn.ModuleList([
            Compute_Block(z_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, 
                                attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(num_compute_layers)
        ])

    def forward(self, z, x):
        z = self.read(z, x)
        for layer in self.compute:
            z = layer(z)
        x = self.write(z, x)
        return z, x
    
def pairwise_distances(x, y):
    #Input: x is a Nxd matrix
    #       y is an optional Mxd matirx
    #Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    #        if y is not given then use 'y=x'.
    #i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def meanshift_cluster(pts_in, bandwidth, weights=None, max_iter=20):
    """
    Meanshift clustering
    :param pts_in: input points
    :param bandwidth: bandwidth
    :param weights: weights per pts indicting its importance in the clustering
    :return: points after clustering
    """
    diff = 1e10
    num_iter = 1
    while diff > 1e-3 and num_iter < max_iter:
        Y = np.sum(((pts_in[np.newaxis, ...] - pts_in[:, np.newaxis, :]) ** 2), axis=2)
        K = np.maximum(bandwidth**2 - Y, np.zeros(Y.shape))
        if weights is not None:
            K = K * weights
        row_sums = K.sum(axis=0, keepdims=True)
        P = K / (row_sums + 1e-10)
        P = P.transpose()
        pts_in_prim = 0.3 * (np.matmul(P, pts_in) - pts_in) + pts_in
        diff = np.sqrt(np.sum((pts_in_prim - pts_in)**2))
        pts_in = pts_in_prim
        num_iter += 1
    return pts_in

def nms_meanshift(pts_in, density, bandwidth):
    """
    NMS to extract modes after meanshift. Code refers to sci-kit-learn.
    :param pts_in: input points
    :param density: density at each point
    :param bandwidth: bandwidth used in meanshift. Used here as neighbor region for NMS
    :return: extracted clusters.
    """
    Y = np.sum(((pts_in[np.newaxis, ...] - pts_in[:, np.newaxis, :]) ** 2), axis=2)
    sorted_ids = np.argsort(density)[::-1]
    unique = np.ones(len(sorted_ids), dtype=bool)
    dist = np.sqrt(Y)
    for i in sorted_ids:
        if unique[i]:
            neighbor_idxs = np.argwhere(dist[:, i] <= bandwidth)
            unique[neighbor_idxs.squeeze()] = 0
            unique[i] = 1  # leave the current point as unique
    pts_in = pts_in[unique]
    return pts_in

def get_predictions(y_pred_np, attn_pred_np=None,bandwidth=0.05, threshold=0.001):
    """
    get the final predictions
    :param pts: input points
    :param weights: weight per point during clustering
    :return: clustered points
    """
    # if attn_pred_np is None:
    #     attn_pred_np = np.ones(y_pred_np.shape[0])
    y_pred_np = meanshift_cluster(y_pred_np, bandwidth, attn_pred_np, max_iter=40)

    
    Y_dist = np.sum(((y_pred_np[np.newaxis, ...] - y_pred_np[:, np.newaxis, :]) ** 2), axis=2)
    density = np.maximum(bandwidth ** 2 - Y_dist, np.zeros(Y_dist.shape))
    density = np.sum(density, axis=0)
    density_sum = np.sum(density)
    y_pred_np = y_pred_np[density / density_sum > threshold]

    density = density[density / density_sum > threshold]
    pred_joints = nms_meanshift(y_pred_np, density, bandwidth)
    return pred_joints

        
if __name__ == '__main__':
    points_cloud = np.ones((100, 3))
    predict_out = get_predictions(points_cloud, bandwidth=0.05, threshold=0.001)
    print(predict_out.shape)
    