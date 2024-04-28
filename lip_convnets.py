import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from skew_ortho_conv import SOC, MonarchSOC

from custom_activations import *
from utils import conv_mapping, activation_mapping

class NormalizedLinear(nn.Linear):
    def forward(self, X):
        X = X.view(X.shape[0], -1)
        weight_norm = torch.norm(self.weight, dim=1, keepdim=True)
        self.lln_weight = self.weight/weight_norm
        return F.linear(X, self.lln_weight if self.training else self.lln_weight.detach(), self.bias)

class LipBlock(nn.Module):
    def __init__(self, in_planes, planes, conv_layer, activation_name, stride=1, kernel_size=3, groups=1):
        super(LipBlock, self).__init__()
        self.conv = conv_layer(in_planes, planes*stride, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2, groups=groups)
        self.activation = activation_mapping(activation_name, planes*stride)

    def forward(self, x):
        x = self.activation(self.conv(x))
        return x
        
class LipConvNet(nn.Module):
    def __init__(self, conv_name, activation, init_channels=32, block_size=1, 
                 num_classes=10, input_side=32, lln=False, groups=1):
        super(LipConvNet, self).__init__()        
        self.lln = lln
        self.in_planes = 3
        
        conv_layer = conv_mapping[conv_name]
        assert type(block_size) == int

        self.layer1 = self._make_layer(init_channels, block_size, conv_layer, 
                                       activation, stride=2, kernel_size=3, groups=groups)
        self.layer2 = self._make_layer(self.in_planes, block_size, conv_layer, 
                                       activation, stride=2, kernel_size=3, groups=groups)
        self.layer3 = self._make_layer(self.in_planes, block_size, conv_layer, 
                                       activation, stride=2, kernel_size=3, groups=groups)
        self.layer4 = self._make_layer(self.in_planes, block_size, conv_layer,
                                       activation, stride=2, kernel_size=3, groups=groups)
        self.layer5 = self._make_layer(self.in_planes, block_size, conv_layer, 
                                       activation, stride=2, kernel_size=1, groups=groups)
        
        flat_size = input_side // 32
        flat_features = flat_size * flat_size * self.in_planes
        if self.lln:
            self.last_layer = NormalizedLinear(flat_features, num_classes)
        else:
            self.last_layer = conv_layer(flat_features, num_classes, 
                                         kernel_size=1, stride=1)

    def _make_layer(self, planes, num_blocks, conv_layer, activation, 
                    stride, kernel_size, groups):
        strides = [1]*(num_blocks-1) + [stride]
        kernel_sizes = [3]*(num_blocks-1) + [kernel_size]
        layers = []
        for stride, kernel_size in zip(strides, kernel_sizes):
            layers.append(LipBlock(self.in_planes, planes, conv_layer, activation, 
                                   stride, kernel_size, groups))
            self.in_planes = planes * stride
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.last_layer(x)
        x = x.view(x.shape[0], -1)
        return x