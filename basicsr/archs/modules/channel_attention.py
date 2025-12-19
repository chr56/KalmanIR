from typing import Literal

import torch

from torch import nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module from 'Squeeze-and-Excitation Networks'
    """

    def __init__(
            self,
            channel: int,
            squeezed_channel: int = -1,
            leaky_relu: float = 0,
            pooling: Literal['avg', 'max'] = 'avg',
    ):
        super(ChannelAttention, self).__init__()

        if squeezed_channel <= 0:
            squeezed_channel = channel // 16

        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError('pooling must be "avg" or "max"')

        self.conv1 = nn.Conv2d(channel, squeezed_channel, 1, padding=0)
        self.activation = nn.LeakyReLU(leaky_relu, inplace=True) if leaky_relu > 0 else nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeezed_channel, channel, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.conv1(y)
        y = self.activation(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y


class ChannelCrossAttention(nn.Module):
    def __init__(
            self,
            channel: int,
            squeezed_channel: int = -1,
            leaky_relu: float = 0,
            pooling: Literal['avg', 'max'] = 'avg',
            channel_last: bool = False,
    ):
        super(ChannelCrossAttention, self).__init__()

        if squeezed_channel <= 0:
            squeezed_channel = channel // 16

        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError('pooling must be "avg" or "max"')

        self.conv1 = nn.Conv2d(channel, squeezed_channel, 1, padding=0)
        self.activation = nn.LeakyReLU(leaky_relu, inplace=True) if leaky_relu > 0 else nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeezed_channel, channel, 1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.channel_last = channel_last

    def forward(self, x, ref):
        if self.channel_last:
            ref = ref.permute(0, 3, 1, 2).contiguous()

        y = self.pool(ref)

        y = self.conv1(y)
        y = self.activation(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        if self.channel_last:
            y = y.permute(0, 2, 3, 1).contiguous()

        return x * y


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from 'CBAM: Convolutional Block Attention Module'
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class BottleneckConv(nn.Module):
    def __init__(self, channel: int, compressed_channel: int = -1, channel_last: bool = False):
        super().__init__()

        if compressed_channel <= 0:
            compressed_channel = channel // 3

        self.conv1 = nn.Conv2d(channel, compressed_channel, kernel_size=3, stride=1, padding=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(compressed_channel, channel, kernel_size=3, stride=1, padding=1)
        self.channel_last = channel_last

    def forward(self, x):
        if self.channel_last:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.channel_last:
            x = x.permute(0, 2, 3, 1).contiguous()
        return x


class ChannelAttentionBlock(nn.Module):
    """Channel Attention Block"""

    def __init__(
            self,
            channel: int,
            leaky_relu: float = 0,
            pooling: Literal['avg', 'max'] = 'avg',
            compressed_channel: int = -1,
            squeezed_channel: int = -1,
            channel_last: bool = True,
    ):
        super(ChannelAttentionBlock, self).__init__()
        self.channel_last = channel_last
        self.bottleneck_conv = BottleneckConv(channel, compressed_channel)
        self.channel_attention = ChannelAttention(
            channel, squeezed_channel, leaky_relu=leaky_relu, pooling=pooling,
        )

    def forward(self, x):
        if self.channel_last:
            x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bottleneck_conv(x)
        x = self.channel_attention(x)

        if self.channel_last:
            x = x.permute(0, 2, 3, 1).contiguous()

        return x
