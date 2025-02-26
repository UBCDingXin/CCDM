### based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

import os
import math
from abc import abstractmethod
from functools import partial
from PIL import Image
import requests
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange



def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern = None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse

# classifier free guidance functions
def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def project(x, y):
    x, inverse = pack_one_with_inverse(x, 'b *')
    y, _ = pack_one_with_inverse(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.view(-1)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, cond_emb_dim = 0):
        super().__init__()
        self.cond_emb_dim = cond_emb_dim
        self.tc_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(cond_emb_dim), dim_out * 2)
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, cond_emb = None):

        scale_shift = None
        
        # Add time step and condition embeddings
        if cond_emb is not None:
            assert self.cond_emb_dim>0
            tc_emb = tuple((time_emb, cond_emb))
            tc_emb = torch.cat(tc_emb, dim = 1)
        else:
            assert self.cond_emb_dim==0
            tc_emb = time_emb
        tc_emb = self.tc_mlp(tc_emb)
        tc_emb = rearrange(tc_emb, 'b c -> b c 1 1')
        scale_shift = tc_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        embed_input_dim=128, #embedding dim of regression label
        cond_drop_prob = 0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        in_channels = 3,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        super().__init__()

        self.in_channels = in_channels

        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob

        # determine dimensions
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(in_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # condition embeddings
        
        self.cond_mlp_1 = nn.Sequential(
            nn.Linear(embed_input_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )
        
        ## null_cond_emb in (https://github.com/lucidrains/denoising-diffusion-pytorch/blob/1d9d8dffb72e02172da8a77bee039b1c72b7c6d5/denoising_diffusion_pytorch/classifier_free_guidance.py#L330) is defined as torch.randn with requires_grad=True
        self.null_cond_emb = nn.Parameter(-1*torch.abs(torch.randn(dim)), requires_grad=True)
        
        cond_emb_dim = dim * 4
        self.cond_mlp_2 = nn.Sequential(
            nn.Linear(dim, cond_emb_dim),
            nn.BatchNorm1d(cond_emb_dim),
            nn.ReLU(),
        )
               
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, cond_emb_dim = cond_emb_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, cond_emb_dim = cond_emb_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, cond_emb_dim = cond_emb_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, cond_emb_dim = cond_emb_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim, cond_emb_dim = cond_emb_dim),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim, cond_emb_dim = cond_emb_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = in_channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim = time_dim, cond_emb_dim = cond_emb_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        rescaled_phi = 0.,
        remove_parallel_component = True,
        keep_parallel_frac = 0.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        update = logits - null_logits

        if remove_parallel_component:
            parallel, orthog = project(update, logits)
            update = orthog + parallel * keep_parallel_frac

        scaled_logits = logits + update * (cond_scale - 1.)

        if rescaled_phi == 0.:
            return scaled_logits, null_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        interpolated_rescaled_logits = rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

        return interpolated_rescaled_logits, null_logits

    def forward(
        self,
        x,
        timesteps,
        labels_emb,
        cond_drop_prob = None,
        keep_mask = None,
        return_bottleneck=False,
    ):
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance        

        c_emb = self.cond_mlp_1(labels_emb)

        if cond_drop_prob > 0:
            
            if keep_mask is not None and cond_drop_prob is None: # if keep_mask is given use this given mask
                self.keep_mask = keep_mask
            else:
                self.keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)               
                
            null_cond_emb = repeat(self.null_cond_emb, 'd -> b d', b = batch)

            c_emb = torch.where(
                rearrange(self.keep_mask, 'b -> b 1'),
                c_emb,
                null_cond_emb
            )
            
        c_emb = self.cond_mlp_2(c_emb)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t_emb = self.time_mlp(timesteps)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_emb, c_emb)
            h.append(x)

            x = block2(x, t_emb, c_emb)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t_emb, c_emb)
        if return_bottleneck:
            return x
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb, c_emb)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t_emb, c_emb)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t_emb, c_emb)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t_emb, c_emb)
        
        return self.final_conv(x)
    



if __name__ == "__main__":
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    NC = 3
    # IMG_SIZE = 192
    # DIM_MULTS=(1, 2, 2, 4, 4, 8, 8) #hiddem_dim = 512, 3, 3
    IMG_SIZE = 128
    DIM_MULTS=(1, 2, 4, 4, 8, 8) #hiddem_dim = 512, 4, 4
    # IMG_SIZE = 64
    # DIM_MULTS=(1, 2, 4, 4, 8) #hiddem_dim = 512, 4, 4
    # IMG_SIZE = 64
    # DIM_MULTS=(1, 2, 2, 4) #hiddem_dim = 256, 8, 8
    
    embed_input_dim = 128
    timesteps = 100

    model = Unet(
        dim=64,
        embed_input_dim=128, #embedding dim of regression label
        cond_drop_prob = 0.5,
        dim_mults=DIM_MULTS,
        in_channels = NC,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attn_dim_head = 32,
        attn_heads = 4
    )
    model = nn.DataParallel(model)

    N=4
    x = torch.randn(N, NC, IMG_SIZE, IMG_SIZE).cuda()
    t = torch.randint(0,timesteps, size=(N,1))
    c = torch.randn(N, embed_input_dim).cuda()

    x_hat = model(x, t, c, return_bottleneck=True)
    
    print(x_hat.size())

    print('model size:', get_parameter_number(model))