'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable


cfg = {
    'VGG8':  [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

NC=1

class vgg(nn.Module):
    def __init__(self, vgg_name):
        super(vgg, self).__init__()
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
        in_channels = NC
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

def vgg8():
    model = vgg('VGG8')
    return model


def vgg11():
    model = vgg('VGG11')
    return model

def vgg13():
    model = vgg('VGG13')
    return model

def vgg16():
    model = vgg('VGG16')
    return model

def vgg19():
    model = vgg('VGG19')
    return model


if __name__ == "__main__":
    net = vgg8().cuda()
    net = nn.DataParallel(net)
    x = torch.randn(4,3,64,64)
    print(net(x).size())
