from typing import Callable, Literal, Union

import torch
from torch import nn


# borrow from KEEP: https://github.com/jnjaby/KEEP
class ConvolutionalResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels=None,
            norm_num_groups_1=None,
            norm_num_groups_2=None,
    ):
        super(ConvolutionalResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_groups_1 = in_channels // 4 if norm_num_groups_1 is None else norm_num_groups_1
        self.norm_groups_2 = out_channels // 4 if norm_num_groups_2 is None else norm_num_groups_2

        self.norm1 = nn.GroupNorm(num_groups=self.norm_groups_1, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=self.norm_groups_2, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)

        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


ActivationFunction = Literal['relu', 'leaky_relu', 'silu', 'sigmoid']


def get_activation_function(activation_type, required: bool = True, **kwargs):
    if activation_type == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_type == 'leaky_relu':
        return nn.LeakyReLU(inplace=True, **kwargs)
    elif activation_type == 'silu':
        return nn.SiLU(inplace=True)
    elif activation_type == 'sigmoid':
        return nn.Sigmoid()
    else:
        if required:
            raise NotImplementedError(f"activation type `{activation_type}` is unimplemented")
        else:
            return nn.Identity()


class ConvolutionalResBlockLayerNorm(nn.Module):
    def __init__(
            self,
            channels: int,
            out_channels: int = -1,
            activation_type: ActivationFunction = 'relu',
            kernel_size: int = 3
    ):
        super(ConvolutionalResBlockLayerNorm, self).__init__()

        out_channels = out_channels if out_channels > 0 else channels

        self.channels = channels
        self.out_channels = out_channels

        from .utils import LayerNorm2d
        self.norm1 = LayerNorm2d(channels, eps=1e-6, elementwise_affine=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding='same', stride=1)
        self.norm2 = LayerNorm2d(channels, eps=1e-6, elementwise_affine=True)
        self.conv2 = nn.Conv2d(channels, out_channels, kernel_size=kernel_size, padding='same', stride=1)

        if channels != out_channels:
            self.conv_residual = nn.Conv2d(channels, out_channels, kernel_size=1, padding='same')

        self.activation = get_activation_function(activation_type)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        if self.channels != self.out_channels:
            x_in = self.conv_residual(x_in)

        return x + x_in


class ConvolutionalResBlockGroupNorm(nn.Module):
    def __init__(
            self,
            channels: int,
            norm_group: int = 1,
            out_channels: int = -1,
            activation_type: ActivationFunction = 'relu',
            kernel_size: int = 3
    ):
        super(ConvolutionalResBlockGroupNorm, self).__init__()

        out_channels = out_channels if out_channels > 0 else channels

        self.channels = channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(norm_group, channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding='same', stride=1)
        self.norm2 = nn.GroupNorm(norm_group, channels, eps=1e-6)
        self.conv2 = nn.Conv2d(channels, out_channels, kernel_size=kernel_size, padding='same', stride=1)

        if channels != out_channels:
            self.conv_residual = nn.Conv2d(channels, out_channels, kernel_size=1, padding='same')

        self.activation = get_activation_function(activation_type)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        if self.channels != self.out_channels:
            x_in = self.conv_residual(x_in)

        return x + x_in
