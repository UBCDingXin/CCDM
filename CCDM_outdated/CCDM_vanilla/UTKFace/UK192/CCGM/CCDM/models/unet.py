import os
import math
from abc import abstractmethod

from PIL import Image
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

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


# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)   
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


# define TimestepEmbedSequential to support `time_emb` as extra input
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, t_emb, c_emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, t_emb, c_emb=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t_emb, c_emb)
            else:
                x = layer(x)
        return x

# use GN for norm layer
def norm_layer(channels, num_groups=32):
    return nn.GroupNorm(num_groups, channels)
    # return nn.BatchNorm2d(channels)


# Residual block
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, cond_channels, dropout, use_scale_shift_norm=False, num_groups=32):

        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        self.cond_channels = cond_channels
        
        self.conv1 = nn.Sequential(
            norm_layer(in_channels, num_groups=num_groups),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # pojection for timestep embedding
        self.tc_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_channels) + int(cond_channels), 2 * out_channels if use_scale_shift_norm else out_channels)
        )
        
        self.conv2 = nn.Sequential(
            norm_layer(out_channels, num_groups=num_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x, time_emb, cond_emb=None):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `time_embed` has shape `[batch_size, time_dim]`
        `cond_embed` has shape `[batch_size, cond_dim]`
        """
        h = self.conv1(x)

        # Add time step and condition embeddings
        if cond_emb is not None:
            assert self.cond_channels>0
            tc_emb = tuple((time_emb, cond_emb))
            tc_emb = torch.cat(tc_emb, dim = 1)
        else:
            assert self.cond_channels==0
            tc_emb = time_emb
        tc_emb = self.tc_mlp(tc_emb)
        tc_emb = rearrange(tc_emb, 'b c -> b c 1 1')
        
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(tc_emb, 2, dim=1)
            h = self.conv2[0](h) * (1 + scale) + shift # GroupNorm(h)(w+1) + b
            h = self.conv2[1:](h)
        else:
            h = self.conv2(h + tc_emb)
        return h + self.shortcut(x)

# Attention block with shortcut
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_groups=32):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = norm_layer(channels, num_groups=num_groups)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


# upsample
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

# downsample
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)


# The full UNet model with attention and timestep embedding
class Unet(nn.Module):
    def __init__(
        self,
        embed_input_dim=128, #embedding dim of regression label
        cond_drop_prob = 0.5,
        in_channels=3,
        model_channels=128,
        out_channels=None, 
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 4, 8), 
        conv_resample=True,
        num_heads=4,
        use_scale_shift_norm=True,
        learned_variance = False,
        num_groups=32,
    ):
        super().__init__()

        default_out_dim = in_channels * (1 if not learned_variance else 2)
        out_channels = default(out_channels, default_out_dim)

        self.embed_input_dim = embed_input_dim
        self.cond_drop_prob = cond_drop_prob

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_groups = num_groups
        
        # time embedding
        time_embed_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # condition embeddings
        cond_embed_dim = model_channels * 4
        self.classes_emb = nn.Sequential(
            nn.Linear(embed_input_dim, cond_embed_dim),
            nn.BatchNorm1d(cond_embed_dim),
            nn.ReLU(),
        )
        # self.null_classes_emb = nn.Parameter(torch.randn(cond_embed_dim))
        # self.null_classes_emb = nn.Parameter(-1*torch.ones(cond_embed_dim), requires_grad=False)
        self.null_classes_emb = nn.Parameter(-1*torch.abs(torch.randn(cond_embed_dim)), requires_grad=False)
        # self.null_classes_emb = nn.Parameter(torch.zeros(cond_embed_dim)-(1e-6), requires_grad=False)
        # self.cond_mlp = nn.Sequential(
        #     nn.Linear(cond_embed_dim, cond_embed_dim),
        #     nn.GELU(),
        #     nn.Linear(cond_embed_dim, cond_embed_dim)
        # )
        
        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, cond_embed_dim, dropout,
                        use_scale_shift_norm=use_scale_shift_norm, num_groups=num_groups)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, num_groups=num_groups))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1: # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2
        
        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, cond_channels=0, dropout=dropout, use_scale_shift_norm=use_scale_shift_norm, num_groups=num_groups),
            AttentionBlock(ch, num_heads=num_heads, num_groups=num_groups),
            ResidualBlock(ch, ch, time_embed_dim, cond_channels=0, dropout=dropout, use_scale_shift_norm=use_scale_shift_norm, num_groups=num_groups)
        )
        
        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        cond_embed_dim,
                        dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=num_groups
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, num_groups=num_groups))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch, num_groups=num_groups),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, 
                x, 
                timesteps, 
                classes,
                cond_drop_prob = None,
                return_null_indx = False):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        batch, device = x.shape[0], x.device

        hs = []

        # time step embedding
        t_emb = self.time_mlp(timestep_embedding(timesteps, self.model_channels))
        t_emb = torch.squeeze(t_emb)

        # derive condition, with condition dropout for classifier free guidance    
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        c_emb = self.classes_emb(classes)
        c_emb = torch.squeeze(c_emb)
        if cond_drop_prob > 0:
            self.keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
            c_emb = torch.where(
                rearrange(self.keep_mask, 'b -> b 1'),
                c_emb, #True
                null_classes_emb #False
            )
        # c_emb = self.cond_mlp(c_emb)
        
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h, t_emb, c_emb)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, t_emb, None)

        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, t_emb, c_emb)

        if return_null_indx:
            null_indx = torch.where(self.keep_mask==False)[0]
            return self.out(h), null_indx
        else:
            return self.out(h)




if __name__ == "__main__":
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    embed_input_dim = 128
    timesteps = 100

    model = Unet(
        embed_input_dim = embed_input_dim,
        cond_drop_prob = 0.5,
        in_channels=3,
        model_channels=64,
        out_channels=None,
        num_res_blocks=2,
        attention_resolutions=(12, 24, 48, 96),
        dropout=0,
        channel_mult=(1, 2, 2, 4, 4, 8, 8), 
        conv_resample=True,
        num_heads=4,
        use_scale_shift_norm=True,
        learned_variance = False,
        )
    model = nn.DataParallel(model)

    N=4
    x = torch.randn(N, 3, 192, 192).cuda()
    t = torch.randint(0,timesteps, size=(N,1))
    c = torch.randn(N, embed_input_dim).cuda()

    x_hat = model(x, t, c)
    
    print(x_hat.size())

    print('model size:', get_parameter_number(model))