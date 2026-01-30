'''
The network for Spectral Normalization GAN (SNGAN).

https://github.com/christiancosgrove/pytorch-spectral-normalization-gan

chainer: https://github.com/pfnet-research/sngan_projection
'''

# ResNet generator and discriminator
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from torch.nn.utils import spectral_norm

########################
# generator

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, dim_embed):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        self.embed_gamma = nn.Linear(dim_embed, num_features, bias=False)
        self.embed_beta = nn.Linear(dim_embed, num_features, bias=False)

    def forward(self, x, y):
        out = self.bn(x)

        gamma = self.embed_gamma(y).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(y).view(-1, self.num_features, 1, 1)
        out = out + out*gamma + beta

        return out


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, dim_embed, bias=True, do_upsample=True):
        super(ResBlockGenerator, self).__init__()

        self.do_upsample = do_upsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=bias)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.condbn1 = ConditionalBatchNorm2d(in_channels, dim_embed)
        self.condbn2 = ConditionalBatchNorm2d(out_channels, dim_embed)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)

        # unconditional case
        if self.do_upsample:
            self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                self.conv1,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                self.conv2
                )
        else:
            self.model = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                self.conv1,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                self.conv2
                )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=bias) #h=h
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        if self.do_upsample:
            self.bypass = nn.Sequential(
                nn.Upsample(scale_factor=2),
                self.bypass_conv,
            )
        else:
            self.bypass = self.bypass_conv

    def forward(self, x, y):
        if y is not None:
            out = self.condbn1(x, y)
            out = self.relu(out)
            if self.do_upsample:
                out = self.upsample(out)
            out = self.conv1(out)
            out = self.condbn2(out, y)
            out = self.relu(out)
            out = self.conv2(out)
            out = out + self.bypass(x)
        else:
            out = self.model(x) + self.bypass(x)

        return out


class sngan_generator(nn.Module):
    
    def __init__(self, dim_z=128, dim_y=128, nc=3, img_size=64, gene_ch=32, ch_multi=None):
        super(sngan_generator, self).__init__()
        self.dim_z = dim_z
        self.dim_embed = dim_y
        self.gene_ch = gene_ch
        self.img_size = img_size
        assert self.img_size in [64, 128, 192, 256]

        if self.img_size == 64:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[16, 8, 8, 4, 2, 1]
            assert len(ch_multi)>=5
        elif self.img_size == 128:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[16, 8, 8, 4, 4, 2, 1]
            assert len(ch_multi)>=6    
        elif self.img_size == 192:
            self.init_size = 3
            if ch_multi is None:
                ch_multi=[16, 8, 8, 4, 4, 2, 1]
            assert len(ch_multi)>=7
        elif self.img_size == 256:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[16, 8, 8, 4, 4, 2, 1]
            assert len(ch_multi)>=7
        self.ch_multi = ch_multi

        self.dense = nn.Linear(self.dim_z, self.init_size * self.init_size * gene_ch*ch_multi[0], bias=True)
        self.final = nn.Conv2d(gene_ch, nc, 3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.genblock0 = ResBlockGenerator(gene_ch*ch_multi[0], gene_ch*ch_multi[1], dim_embed=self.dim_embed) #4--->8, or 3--->6
        self.genblock1 = ResBlockGenerator(gene_ch*ch_multi[1], gene_ch*ch_multi[2], dim_embed=self.dim_embed) #8--->16, or 6--->12
        self.genblock2 = ResBlockGenerator(gene_ch*ch_multi[2], gene_ch*ch_multi[3], dim_embed=self.dim_embed) #16--->32, or 12--->24
        self.genblock3 = ResBlockGenerator(gene_ch*ch_multi[3], gene_ch*ch_multi[4], dim_embed=self.dim_embed) #32--->64, or 24--->48
        if self.img_size in [64]:
            self.genblock4 = ResBlockGenerator(gene_ch*ch_multi[4], gene_ch*ch_multi[-1], dim_embed=self.dim_embed, do_upsample=False) #64-->64
        if self.img_size in [128]:
            self.genblock4 = ResBlockGenerator(gene_ch*ch_multi[4], gene_ch*ch_multi[5], dim_embed=self.dim_embed) #64--->128
            self.genblock5 = ResBlockGenerator(gene_ch*ch_multi[5], gene_ch*ch_multi[-1], dim_embed=self.dim_embed, do_upsample=False) #128--->128
        if self.img_size in [192, 256]:
            self.genblock4 = ResBlockGenerator(gene_ch*ch_multi[4], gene_ch*ch_multi[5], dim_embed=self.dim_embed) #64--->128, or 48--->96
            self.genblock5 = ResBlockGenerator(gene_ch*ch_multi[5], gene_ch*ch_multi[-1], dim_embed=self.dim_embed) #128--->256, or 96--->192

        self.final = nn.Sequential(
            nn.BatchNorm2d(gene_ch*ch_multi[-1]),
            nn.ReLU(),
            self.final,
            nn.Tanh()
        )

    def forward(self, z, y): #y is embedded in the feature space
        z = z.view(z.size(0), z.size(1))
        out = self.dense(z)
        out = out.view(-1, self.gene_ch*self.ch_multi[0], self.init_size, self.init_size)

        out = self.genblock0(out, y)
        out = self.genblock1(out, y)
        out = self.genblock2(out, y)
        out = self.genblock3(out, y)
        if self.img_size in [64, 128, 192, 256]:
            out = self.genblock4(out, y)
        if self.img_size in [128, 256, 192]:
            out = self.genblock5(out, y)
        out = self.final(out)

        return out




########################
# discriminator

class ResBlockDiscriminator(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )

        self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        if stride != 1:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        else:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=True)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class sngan_discriminator(nn.Module):
    
    def __init__(self, dim_y=128, nc=3, img_size=64, disc_ch=32, ch_multi=None, use_aux_reg=False, use_aux_dre=False, dre_head_arch="MLP3", p_dropout=0.5):
        
        super(sngan_discriminator, self).__init__()
        self.dim_embed = dim_y
        self.disc_ch = disc_ch
        
        self.nc = nc
        self.img_size = img_size
        assert self.img_size in [64, 128, 192, 256]
        
        self.use_aux_reg = use_aux_reg
        self.use_aux_dre = use_aux_dre
        self.dre_head_arch = dre_head_arch
        
        if self.img_size == 64:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[1, 2, 4, 8, 16]
            assert len(ch_multi)>=5
        elif self.img_size == 128:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[1, 2, 2, 4, 8, 16]
            assert len(ch_multi)>=6    
        elif self.img_size == 192:
            self.init_size = 3
            if ch_multi is None:
                ch_multi=[1, 2, 2, 4, 8, 8, 16]
            assert len(ch_multi)>=7
        elif self.img_size == 256:
            self.init_size = 4
            if ch_multi is None:
                ch_multi=[1, 2, 4, 4, 8, 8, 16]
            assert len(ch_multi)>=7
        self.ch_multi = ch_multi

        self.discblock1 = nn.Sequential(
            FirstResBlockDiscriminator(nc, disc_ch*ch_multi[0], stride=2), #64--->32, 128--->64, 192--->96, 256--->128
            ResBlockDiscriminator(disc_ch*ch_multi[0], disc_ch*ch_multi[1], stride=2), #32--->16, 64--->32, 96--->48, 128--->64
            ResBlockDiscriminator(disc_ch*ch_multi[1], disc_ch*ch_multi[2], stride=2), #16--->8, 32--->16, 48--->24, 64--->32
            ResBlockDiscriminator(disc_ch*ch_multi[2], disc_ch*ch_multi[3], stride=2) #8--->4, 16--->8, 24--->12, 32--->16
        )
        
        if self.img_size in [64]:
            self.discblock2 = nn.Sequential(
                ResBlockDiscriminator(disc_ch*ch_multi[3], disc_ch*ch_multi[-1], stride=1), #4--->4
                nn.ReLU(),
            )
        elif self.img_size in [128]:
            self.discblock2 = nn.Sequential(
                ResBlockDiscriminator(disc_ch*ch_multi[3], disc_ch*ch_multi[4], stride=2), #8--->4
                ResBlockDiscriminator(disc_ch*ch_multi[4], disc_ch*ch_multi[-1], stride=1), #4--->4
                nn.ReLU(),
            )
        else:
            self.discblock2 = nn.Sequential(
                ResBlockDiscriminator(disc_ch*ch_multi[3], disc_ch*ch_multi[4], stride=2), #16--->8, or 12--->6
                ResBlockDiscriminator(disc_ch*ch_multi[4], disc_ch*ch_multi[5], stride=2), #8--->4, or 6--->3
                ResBlockDiscriminator(disc_ch*ch_multi[5], disc_ch*ch_multi[-1], stride=1), #4--->4, or 3--->3
                nn.ReLU(),
            )

        self.linear1 = nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size, 1, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        self.linear1 = spectral_norm(self.linear1)
        self.linear2 = nn.Linear(self.dim_embed, disc_ch*self.ch_multi[-1]*self.init_size*self.init_size, bias=False)
        nn.init.xavier_uniform_(self.linear2.weight.data, 1.)
        self.linear2 = spectral_norm(self.linear2)
        
        if use_aux_reg:
            self.reg_linear = nn.Sequential(
                spectral_norm(nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size, 128)),
                nn.ReLU(),
                spectral_norm(nn.Linear(128, 1)),
                nn.ReLU(),
            )
        
        if use_aux_dre:
            if self.dre_head_arch == "MLP5":
                self.dre_linear = nn.Sequential(
                    nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size+self.dim_embed, 2048),
                    nn.GroupNorm(8, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 1024),
                    nn.GroupNorm(8, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.GroupNorm(8, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.GroupNorm(8, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.ReLU(),
                )
            elif self.dre_head_arch == "MLP5_dropout":
                self.dre_linear = nn.Sequential(
                    nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size+self.dim_embed, 2048),
                    nn.GroupNorm(8, 2048),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(2048, 1024),
                    nn.GroupNorm(8, 1024),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(1024, 512),
                    nn.GroupNorm(8, 512),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(512, 256),
                    nn.GroupNorm(8, 256),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(256, 1),
                    nn.ReLU(),
                )
            elif self.dre_head_arch == "MLP3_dropout":
                self.dre_linear = nn.Sequential(               
                    nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size+self.dim_embed, 512),
                    nn.GroupNorm(8, 512),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(512, 256),
                    nn.GroupNorm(8, 256),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Linear(256, 1),
                    nn.ReLU(),
                )
            elif self.dre_head_arch == "MLP3":
                self.dre_linear = nn.Sequential(               
                    nn.Linear(disc_ch*self.ch_multi[-1]*self.init_size*self.init_size+self.dim_embed, 512),
                    nn.GroupNorm(8, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.GroupNorm(8, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.ReLU(),
                )
            else:
                raise ValueError("Not Supported DRE Branch!")
            
    def forward(self, x, y):
        h = self.discblock1(x)
        h = self.discblock2(h)
        h = h.view(-1, self.disc_ch*self.ch_multi[-1]*self.init_size*self.init_size)
        
        output_y = torch.sum(h*self.linear2(y), 1, keepdim=True)
        adv_output = (self.linear1(h) + output_y).view(-1,1)
        
        # auxiliary regression branch
        reg_output = None
        if self.use_aux_reg:
            reg_output = self.reg_linear(h).view(-1,1)
            
        # auxiliary dre branch
        dre_output = None
        if self.use_aux_dre:
            feat_cat = torch.cat((h, y), -1)
            dre_output = self.dre_linear(feat_cat).view(-1,1)

        return {
            "h":h,
            "adv_output":adv_output,
            "reg_output":reg_output,
            "dre_output":dre_output,
        }



if __name__ == "__main__":
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    IMG_SIZE=128

    netG = sngan_generator(dim_z=256, dim_y=128, nc=3, img_size=IMG_SIZE, gene_ch=64).cuda() # parameters
    netD = sngan_discriminator(dim_y=128, nc=3, img_size=IMG_SIZE, disc_ch=48, use_aux_reg=True, use_aux_dre=True, dre_head_arch="MLP3").cuda() # parameters

    # netG = nn.DataParallel(netG)
    # netD = nn.DataParallel(netD)

    N=4
    z = torch.randn(N, 256).cuda()
    y = torch.randn(N, 128).cuda()
    x = netG(z,y)
    out = netD(x,y)
    print(x.size())
    print(out['adv_output'].size())
    print(out['reg_output'].size())
    print(out['dre_output'].size())
    
    print('G:', get_parameter_number(netG))
    print('D:', get_parameter_number(netD))