'''
ResNet-based model to map an image from pixel space to a features space. For 64x64 images only!
Need to be pretrained on the dataset.

if isometric_map = True, there is an extra step (elf.classifier_1 = nn.Linear(512, 32*32*3)) to increase the dimension of the feature map from 512 to 32*32*3. This selection is for desity-ratio estimation in feature space.

codes are based on
@article{
zhang2018mixup,
title={mixup: Beyond Empirical Risk Minimization},
author={Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz},
journal={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=r1Ddp1-Rb},
}
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_embed_y2cov(nn.Module):
    def __init__(self, block, num_blocks, img_size=64, nc=3):
        super(ResNet_embed_y2cov, self).__init__()
        self.in_planes = 64
        self.dim_embed = img_size**2 * nc
        self.main = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False),  # h=h
            # nn.Conv2d(nc, 64, kernel_size=4, stride=2, padding=1, bias=False),  # h=h/2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # self._make_layer(block, 64, num_blocks[0], stride=1),  # h=h
            self._make_layer(block, 64, num_blocks[0], stride=2),  # h=h/2 32
            self._make_layer(block, 128, num_blocks[1], stride=2), # h=h/2 16
            self._make_layer(block, 256, num_blocks[2], stride=2), # h=h/2 8
            self._make_layer(block, 512, num_blocks[3], stride=2), # h=h/2 4
            # nn.AvgPool2d(kernel_size=4)
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.x2h_res = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            # nn.GroupNorm(32, 512),
            nn.ReLU(),

            nn.Linear(512, self.dim_embed),
            nn.BatchNorm1d(self.dim_embed),
            # nn.GroupNorm(32, dim_embed),
            nn.ReLU(),
            # nn.Sigmoid(),
        )

        self.h2y = nn.Sequential(
            nn.Linear(self.dim_embed, 512),
            nn.BatchNorm1d(512),
            # nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU()
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.main(x)
        features = features.view(features.size(0), -1)
        features = self.x2h_res(features)
        out = self.h2y(features)
        return out, features


def ResNet18_embed_y2cov(img_size, nc):
    return ResNet_embed_y2cov(BasicBlock, [2,2,2,2], img_size=img_size, nc=nc)

def ResNet34_embed_y2cov(img_size, nc):
    return ResNet_embed_y2cov(BasicBlock, [3,4,6,3], img_size=img_size, nc=nc)

def ResNet50_embed_y2cov(img_size, nc):
    return ResNet_embed_y2cov(Bottleneck, [3,4,6,3], img_size=img_size, nc=nc)

#------------------------------------------------------------------------------
# map labels to the embedding space
class model_mlp_y2cov(nn.Module):
    def __init__(self, img_size, nc, num_groups=8):
        super(model_mlp_y2cov, self).__init__()
        dim_embed = img_size**2 * nc
        self.main = nn.Sequential(
            nn.Linear(1, 512),
            # nn.BatchNorm1d(512),
            nn.GroupNorm(num_groups, 512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            # nn.BatchNorm1d(1024),
            nn.GroupNorm(num_groups, 1024),
            nn.ReLU(),

            nn.Linear(1024, 2048),
            # nn.BatchNorm1d(2048),
            nn.GroupNorm(num_groups, 2048),
            nn.ReLU(),

            nn.Linear(2048, 4096),
            # nn.BatchNorm1d(4096),
            nn.GroupNorm(num_groups, 4096),
            nn.ReLU(),

            nn.Linear(4096, dim_embed),
            nn.ReLU()
        )

    def forward(self, y):
        y = y.view(-1, 1)+1e-8
        return self.main(y)


class model_cnn_y2cov(nn.Module):
    def __init__(self, img_size, nc, base_channels=512, num_groups=8):
        super(model_cnn_y2cov, self).__init__()

        super().__init__()
        assert img_size in (64, 128, 192, 256)

        layers = []

        # 1) 1×1 → 4×4 or 3x3
        if img_size in [64,128,256]:
            init_dim = 4
            layers += [
                nn.ConvTranspose2d(1, base_channels, kernel_size=4, stride=1, padding=0, bias=True),
                nn.GroupNorm(num_groups, base_channels),
                # nn.BatchNorm2d(base_channels),
                nn.ReLU(True),
            ]
        else:
            init_dim = 3
            layers += [
                nn.ConvTranspose2d(1, base_channels, kernel_size=5, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, base_channels),
                # nn.BatchNorm2d(base_channels),
                nn.ReLU(True),
            ]

        in_ch = base_channels

        # 2) double the resolution per iteration
        num_iters = int(math.log2(img_size//init_dim))
        for i in range(num_iters):
            layers += [
                nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=4, stride=2, padding=1, bias=True),
                nn.GroupNorm(num_groups, in_ch//2),
                # nn.BatchNorm2d(in_ch//2),
                nn.ReLU(True),
            ]
            in_ch = in_ch//2

        self.layers = nn.Sequential(*layers)

        # 3) Final Conv2d output
        self.final = nn.Conv2d(in_ch, nc, kernel_size=1, stride=1, padding=0)
        self.final_act = nn.ReLU(True)

        self.img_size = img_size

    def forward(self, y):
        y = y.view(-1, 1, 1, 1) + 1e-8
        h = self.layers(y)
        h = self.final(h)
        return self.final_act(h).view(len(y),-1)


if __name__ == "__main__":

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    NC=3
    IMG_SIZE=64
    net = ResNet34_embed_y2cov(img_size=IMG_SIZE, nc=NC).cuda()
    x = torch.randn(16,NC,IMG_SIZE,IMG_SIZE).cuda()
    out, features = net(x)
    print(out.size())
    print(features.size())

    net_y2cov1 = model_mlp_y2cov(img_size=IMG_SIZE, nc=NC).cuda()
    y = torch.randn(16, 1).cuda()
    emb = net_y2cov1(y)
    print(emb.shape)


    net_y2cov2 = model_cnn_y2cov(img_size=IMG_SIZE, nc=NC).cuda()
    y = torch.randn(16, 1).cuda()
    emb = net_y2cov2(y)
    print(emb.shape)

    print('model size:', get_parameter_number(net_y2cov1))
    print('model size:', get_parameter_number(net_y2cov2))
