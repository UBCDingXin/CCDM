"""
Train an auxilary CNN for predicting the regression label of a given 'noisy' image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

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


#######################
## VGG Net
cfg = {
    'VGG8':  [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class vgg(nn.Module):
    def __init__(self, vgg_name, in_channels=3):
        super(vgg, self).__init__()
        self.in_channels = in_channels
        self.features = self._make_layers(cfg[vgg_name])

        self.fc = nn.Sequential(
            nn.Linear(4*4*128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



###################################
# ResNet
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

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(block, 64, num_blocks[0], stride=2),  # h=h/2 32
        )
        self.block2 = self._make_layer(block, 128, num_blocks[1], stride=2) # h=h/2 16
        self.block3 = self._make_layer(block, 256, num_blocks[2], stride=2) # h=h/2 8
        self.block4 = self._make_layer(block, 512, num_blocks[3], stride=2) # h=h/2 4
        self.pool = nn.AvgPool2d(kernel_size=4)

        linear_layers = [
                nn.Linear(512*block.expansion, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                 
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                
                nn.Linear(512, 1),
                nn.ReLU(),
            ]
        self.linear = nn.Sequential(*linear_layers)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        ft1 = self.block1(x)
        ft2 = self.block2(ft1)
        ft3 = self.block3(ft2)
        ft4 = self.block4(ft3)
        out = self.pool(ft4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out




#################################
# define the auxilary network

def make_aux_net(name, in_channels=3):
    if "VGG" in name:
        return vgg(name, in_channels=in_channels)
    elif name=="ResNet18":
        return ResNet(BasicBlock, [2,2,2,2], in_channels=in_channels)
    elif name=="ResNet34":
        return ResNet(BasicBlock, [3,4,6,3], in_channels=in_channels)
    else:
        raise ValueError(f'unknown aux name {name}')




#################################
# debug
if __name__ == "__main__":
    net = make_aux_net("VGG19", in_channels=3).cuda()
    net = nn.DataParallel(net)
    x = torch.randn(4,3,64,64).cuda()
    print(net(x).size())
