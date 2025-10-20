import torch
from torch import nn
from torch.nn import functional as F


class SimpleRecursiveConvolutionalV1(nn.Module):
    def __init__(self, length: int, channel: int, difficult_zone_update_ratio: float):
        super(SimpleRecursiveConvolutionalV1, self).__init__()
        self.iteration = length
        self.channel = channel

        from ..convolutional_res_block import ConvolutionalResBlockLayerNorm

        self.block1 = ConvolutionalResBlockLayerNorm(
            2 * channel, channel, activation_type='leaky_relu'
        )
        self.block2 = ConvolutionalResBlockLayerNorm(
            channel, channel, activation_type='sigmoid'
        )

        self.out_linear = nn.Conv2d(channel, channel, kernel_size=1)

        self.update_ratio = difficult_zone_update_ratio
        self.remain_ratio = 1.0 - difficult_zone_update_ratio

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        s = difficult_zone
        for i in range(self.iteration):
            image = image_sequence[:, i, ...]
            s = torch.cat((s, image), dim=1)
            s = self.block1(s)
            s = self.block2(s)

        uncertainty = s * self.update_ratio + difficult_zone * self.remain_ratio
        uncertainty = self.out_linear(uncertainty)
        return uncertainty


class RecursiveConvolutionalV1(nn.Module):
    def __init__(self, length: int, channel: int, **kwargs):
        super(RecursiveConvolutionalV1, self).__init__()
        self.iteration = length
        self.channel = channel
        self.residual_rate = 0.2
        from ..utils import LayerNorm2d
        from ..convolutional_res_block import ConvolutionalResBlock
        self.norm_image = LayerNorm2d(channel)
        self.conv_block = ConvolutionalResBlock(
            3 * channel, channel,
            norm_num_groups_1=1, norm_num_groups_2=1,
        )

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        assert image_sequence.shape[1] == self.iteration

        uncertainty = difficult_zone
        for i in range(self.iteration):
            image = self.norm_image(image_sequence[:, i, ...])
            block = torch.cat((uncertainty, image, sigma), dim=1)
            uncertainty = self.conv_block(block)

        uncertainty = uncertainty * (1 - self.residual_rate) + difficult_zone * self.residual_rate

        return uncertainty


class RecursiveConvolutionalV2(nn.Module):
    def __init__(self, length: int, channel: int, **kwargs):
        super(RecursiveConvolutionalV2, self).__init__()
        self.iteration = length
        self.channel = channel
        self.residual_rate = 0.2

        self.norm_image = nn.GroupNorm(3, channel)

        self.conv_block_main = nn.Sequential(
            nn.GroupNorm(2, 2 * channel),
            nn.Conv2d(2 * channel, channel, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, channel),
            nn.Conv2d(channel, channel, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
        )
        self.conv_block_side = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size=1, padding='same'),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        assert image_sequence.shape[1] == self.iteration

        uncertainty = difficult_zone
        for i in range(self.iteration):
            image = self.norm_image(image_sequence[:, i, ...])
            block = torch.cat((uncertainty, image), dim=1)
            uncertainty = self.conv_block_main(block) + self.conv_block_side(block)

        uncertainty = uncertainty * (1 - self.residual_rate) + difficult_zone * self.residual_rate

        return uncertainty


class RecursiveConvolutionalV3(nn.Module):
    def __init__(self, length: int, channel: int, **kwargs):
        super(RecursiveConvolutionalV3, self).__init__()
        self.iteration = length
        self.channel = channel

        self.norm_image = nn.GroupNorm(3, channel)

        self.conv_block_main = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, channel),
            nn.Conv2d(channel, channel, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, channel),
            nn.Conv2d(channel, channel, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
        )
        self.conv_block_side = nn.Sequential(
            nn.Conv2d(2 * channel, 2 * channel, kernel_size=3, padding='same'),
            nn.LeakyReLU(inplace=True),
            nn.GroupNorm(2, 2 * channel),
            nn.Conv2d(2 * channel, channel, kernel_size=1, padding='same'),
            nn.LeakyReLU(inplace=True),
        )

        self.merge_rate = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        assert image_sequence.shape[1] == self.iteration

        state = difficult_zone
        for i in range(self.iteration):
            image = self.norm_image(image_sequence[:, i, ...])
            block = torch.cat((state, image), dim=1)
            state = self.conv_block_main(block) + self.conv_block_side(block)

        uncertainty = difficult_zone + state * F.sigmoid(self.merge_rate)
        uncertainty = F.sigmoid(uncertainty)

        return uncertainty
