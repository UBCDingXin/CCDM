'''

Adapted from https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py


'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))



class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out




'''

Generator

'''


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + gamma*out + beta
        return out


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim_embed):
        super(GenBlock, self).__init__()
        self.cond_bn1 = ConditionalBatchNorm2d(in_channels, dim_embed)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = ConditionalBatchNorm2d(out_channels, dim_embed)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, labels, upsample=True):
        x0 = x

        x = self.cond_bn1(x, labels)
        x = self.relu(x)
        if upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.cond_bn2(x, labels)
        x = self.relu(x)
        x = self.snconv2d2(x)

        if upsample:
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class sagan_generator(nn.Module):
    """Generator."""

    def __init__(self, dim_z, dim_embed=128, nc=3, img_size=64, gene_ch=32, ch_multi=None):
        super(sagan_generator, self).__init__()

        self.dim_z = dim_z
        self.gene_ch = gene_ch
        self.img_size = img_size
        assert self.img_size in [64, 128, 192]
        
        if ch_multi is None:
            ch_multi=[16, 8, 4, 2, 1]
        assert len(ch_multi)>=5
        self.ch_multi = ch_multi
        
        if self.img_size in [64,128]:
            self.init_size = 4
        elif self.img_size == 192:
            self.init_size = 6
        
        self.snlinear0 = snlinear(in_features=dim_z, out_features=gene_ch*ch_multi[0]*self.init_size*self.init_size)
        self.block1 = GenBlock(gene_ch*ch_multi[0], gene_ch*ch_multi[1], dim_embed)
        self.block2 = GenBlock(gene_ch*ch_multi[1], gene_ch*ch_multi[2], dim_embed)
        self.self_attn = Self_Attn(gene_ch*ch_multi[2])
        self.block3 = GenBlock(gene_ch*ch_multi[2], gene_ch*ch_multi[3], dim_embed)
        # self.self_attn = Self_Attn(gene_ch*ch_multi[3])
        self.block4 = GenBlock(gene_ch*ch_multi[3], gene_ch*ch_multi[4], dim_embed)
        if self.img_size in [128, 192]:
            self.block5 = GenBlock(gene_ch*ch_multi[4], gene_ch, dim_embed)
        
        self.bn = nn.BatchNorm2d(gene_ch, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=gene_ch, out_channels=nc, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z, labels):
        # labels = self.label_encoder(labels)
        # n x dim_z
        out = self.snlinear0(z)            # self.init_size x self.init_size
        out = out.view(-1, self.gene_ch*self.ch_multi[0], self.init_size, self.init_size) 
        out = self.block1(out, labels)    # 8 x 8 or 12 x 12
        out = self.block2(out, labels)    # 16 x 16 or 24 x 24
        out = self.self_attn(out)         # 16 x 16 or 24 x 24
        out = self.block3(out, labels)    # 32 x 32 or 48 x 48
        # out = self.self_attn(out)         # 32 x 32 or 48 x 48
        out = self.block4(out, labels)    # 64 x 64 or 96 x 96
        if self.img_size in [128, 192]:
            out = self.block5(out, labels)
        out = self.bn(out)
        out = self.relu(out)
        out = self.snconv2d1(out)
        out = self.tanh(out)
        return out



'''

Discriminator

'''

class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class sagan_discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, dim_embed=128, nc=3, img_size=64, disc_ch=32, ch_multi=None):
        super(sagan_discriminator, self).__init__()
        
        self.dim_embed = dim_embed
        
        self.nc = nc
        self.img_size = img_size
        assert self.img_size in [64, 128, 192]
        
        if self.img_size in [64, 128]:
            self.init_size = 4
        else:
            self.init_size = 6
        
        if img_size == 64:
            if ch_multi is None:
                ch_multi=[1, 2, 4, 8, 16]
            assert len(ch_multi)>=5
        else:
            if ch_multi is None:
                ch_multi=[1, 2, 2, 4, 8, 16]
            assert len(ch_multi)>=6
        self.ch_multi = ch_multi
        
        self.disc_ch = disc_ch
        self.opt_block1 = DiscOptBlock(nc, disc_ch*ch_multi[0])
        self.block1 = DiscBlock(disc_ch*ch_multi[0], disc_ch*ch_multi[1])
        self.block2 = DiscBlock(disc_ch*ch_multi[1], disc_ch*ch_multi[2])
        self.block3 = DiscBlock(disc_ch*ch_multi[2], disc_ch*ch_multi[3])
        self.block4 = DiscBlock(disc_ch*ch_multi[3], disc_ch*ch_multi[4])
        
        if self.img_size == 64:
            # self.self_attn = Self_Attn(disc_ch*ch_multi[0])
            self.self_attn = Self_Attn(disc_ch*ch_multi[1])
        else:
            self.self_attn = Self_Attn(disc_ch*ch_multi[1]) 
            self.block5 = DiscBlock(disc_ch*ch_multi[4], disc_ch*ch_multi[5])
        
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=disc_ch*ch_multi[-1]*self.init_size*self.init_size, out_features=1)
        self.sn_embedding1 = snlinear(dim_embed, disc_ch*ch_multi[-1]*self.init_size*self.init_size, bias=False)

        # Weight init
        self.apply(init_weights)
        xavier_uniform_(self.sn_embedding1.weight)

    def forward(self, x, labels):
        
        out = self.opt_block1(x)  
        if self.img_size==64:
            # out = self.self_attn(out) # 32 x 32
            out = self.block1(out)    # 16 x 16
            out = self.self_attn(out) # 16 x 16
            out = self.block2(out)    # 8 x 8
            out = self.block3(out)    # 4 x 4
            out = self.block4(out, downsample=False)    # 4 x 4
        elif self.img_size in [128, 192]:
            out = self.block1(out) # 32 x 32 or 48 x 48
            out = self.self_attn(out) # 32 x 32 or 48 x 48
            out = self.block2(out)    # 16 x 16 or 24 x 24
            out = self.block3(out)    # 8 x 8 or 12 x 12
            out = self.block4(out)    # 4 x 4 or 6 x 6  
            out = self.block5(out, downsample=False)    # 4 x 4 or 6 x 6
        
        out = self.relu(out)              # n x disc_ch*ch_multi[-1] x self.init_size x self.init_size
        out = out.view(-1, self.disc_ch*self.ch_multi[-1]*self.init_size*self.init_size)
        output1 = torch.squeeze(self.snlinear1(out)) # n
        # Projection        
        h_labels = self.sn_embedding1(labels)   # n x disc_ch*self.ch_multi[-1]
        proj = torch.mul(out, h_labels)          # n x disc_ch*self.ch_multi[-1]
        output2 = torch.sum(proj, dim=[1])      # n
        # Out
        output = output1 + output2              # n
        return output.view(-1)


if __name__ == "__main__":
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    IMG_SIZE=192

    netG = sagan_generator(dim_z=256, dim_embed=128, nc=3, img_size=IMG_SIZE, gene_ch=48, ch_multi=[16, 8, 4, 2, 1]).cuda() # parameters
    netD = sagan_discriminator(dim_embed=128, nc=3, img_size=IMG_SIZE, disc_ch=48).cuda() # parameters

    # netG = nn.DataParallel(netG)
    # netD = nn.DataParallel(netD)

    N=4
    z = torch.randn(N, 256).cuda()
    y = torch.randn(N, 128).cuda()
    x = netG(z,y)
    o = netD(x,y)
    print(x.size())
    print(o.size())

    print('G:', get_parameter_number(netG))
    print('D:', get_parameter_number(netD))
