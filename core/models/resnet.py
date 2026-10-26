from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalization(nn.Module):
    """
    Standardizes the input data.
    Arguments:
        mean (list): mean.
        std (float): standard deviation.
        device (str or torch.device): device to be used.
    Returns:
        (input - mean) / std
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        num_channels = len(mean)
        self.mean = torch.FloatTensor(mean).view(1, num_channels, 1, 1)
        self.sigma = torch.FloatTensor(std).view(1, num_channels, 1, 1)
        self.mean_cuda, self.sigma_cuda = None, None

    def forward(self, x):
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.sigma_cuda = self.sigma.cuda()
            out = (x - self.mean_cuda) / self.sigma_cuda
        else:
            out = (x - self.mean) / self.sigma
        return out


class BasicBlock(nn.Module):
    """
    Implements a basic block module for Resnets.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Implements a basic block module with bottleneck for Resnets.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet model
    Arguments:
        block (BasicBlock or Bottleneck): type of basic block to be used.
        num_blocks (list): number of blocks in each sub-module.
        num_classes (int): number of output classes.
        device (torch.device or str): device to work on. 
    """
    def __init__(self, block, num_blocks, num_classes=10, device='cpu',
                 mean: Union[Tuple[float, ...], float] = (0.5, 0.5, 0.5),
                 std: Union[Tuple[float, ...], float] = (0.5, 0.5, 0.5),
                 padding: int = 0,
                 num_input_channels: int = 3,
                 normalize=True,
                 ):
        super(ResNet, self).__init__()
        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda = None
        self.std_cuda = None
        self.normalize = normalize
        self.padding = padding

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.feat_dim = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feats=False):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        if self.normalize:
            if x.is_cuda:
                if self.mean_cuda is None:
                    self.mean_cuda = self.mean.cuda()
                    self.std_cuda = self.std.cuda()
                x = (x - self.mean_cuda) / self.std_cuda
            else:
                x = (x - self.mean) / self.std

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if not feats:
            return self.linear(out)
        else:
            return out, self.linear(out)

    def rf_output(self, x, intermediate_propagate=0, pop=0):
        if intermediate_propagate == 0:
            if self.padding > 0:
                x = F.pad(x, (self.padding,) * 4)
            if self.normalize:
                if x.is_cuda:
                    if self.mean_cuda is None:
                        self.mean_cuda = self.mean.cuda()
                        self.std_cuda = self.std.cuda()
                    x = (x - self.mean_cuda) / self.std_cuda
                else:
                    x = (x - self.mean) / self.std
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            if pop == 1:
                return out
            out = self.layer3(out)
            if pop == 2:
                return out
            out = self.layer4(out)
            if pop == 3:
                return out
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            return self.linear(out)

        elif intermediate_propagate == 1:
            out = x
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            return self.linear(out)

        elif intermediate_propagate == 2:
            out = x
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            return self.linear(out)

        elif intermediate_propagate == 3:
            out = x
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            return self.linear(out)



NUM_BLOCKS = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3],
}

def resnet(name, num_classes=10, normalize=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), device='cpu'):
    """
    Returns suitable Resnet model from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
        pretrained (bool): whether to use a pretrained model.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """

    assert name in NUM_BLOCKS
    name_ = name.replace('resnet', 'ResNet')

    if normalize:
        print(f'{name_} uses normalization {mean}, {std}.')
        return ResNet(BasicBlock, num_blocks=NUM_BLOCKS[name],num_classes=num_classes, device=device,
                      mean=mean, std=std, normalize=True)
    else:
        print(f'{name_}.')
        return ResNet(BasicBlock, num_blocks=NUM_BLOCKS[name],num_classes=num_classes, device=device,
                      normalize=False)
