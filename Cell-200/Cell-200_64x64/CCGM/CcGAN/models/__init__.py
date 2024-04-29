from .dcgan import generator, discriminator
from .ResNet_regre_eval import *
from .ResNet_class_eval import *
from .ResNet_embed import *
from .autoencoder import *
from .vgg import vgg8, vgg11, vgg13, vgg16, vgg19
from .resnetv2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

cnn_dict = {
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'vgg8': vgg8,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
}