from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.refinement import REFINEMENT_ARCH_REGISTRY


@REFINEMENT_ARCH_REGISTRY.register()
class BSRMD(nn.Module):
    """
    MoE Gate from paper 'Blind Single Image Super-resolution with a Mixture of Deep Networks'
    """

    def __init__(
            self,
            branch: int,
            channels: int,
            class_index: int,
            feature_size: int = 32,
            num_conv_layer: int = 3,
            conv_kernel_size: int = 3,
            batch_norm: bool = True,
            **kwargs):
        super(BSRMD, self).__init__()
        self.branch = branch
        self.channels = channels

        self.cls_branch_index = class_index

        self.cls_branch = ClsBranch(
            n_ch_in=self.channels,
            n_ch_out=self.branch,
            n_feat=feature_size,
            kernel_size=conv_kernel_size,
            n_conv=num_conv_layer,
            bn=batch_norm,
        )

    def forward(self, images: List[torch.Tensor]) -> dict:
        # Input shape L * [B, C, H, W]
        x_concat = torch.stack(images, dim=1)  # [B, L, C, H, W]
        weights = self.cls_branch(images[self.cls_branch_index])  # [B, L, 1, 1]
        weights = torch.unsqueeze(weights, dim=-1)  # [B, L, 1, 1, 1]
        averaged = torch.sum(weights * x_concat, dim=1, keepdim=True)  # [B, 1, C, H, W]
        averaged = torch.squeeze(averaged, dim=1)  # [B, C, H, W]

        # Output shape [B, C, H, W]
        return {
            'sr_refined': averaged,  # [B, C, H, W]
        }

    def model_output_format(self):
        return {
            'sr_refined': 'I',
        }

    def primary_output(self):
        return 'sr_refined'


class ClsBranch(nn.Module):
    def __init__(
            self,
            n_ch_in: int = 1,
            n_ch_out: int = 1,
            n_feat: int = 32,
            kernel_size: int = 3,
            n_conv: int = 3,
            bn: bool = True,
    ):
        super(ClsBranch, self).__init__()
        act = nn.LeakyReLU(0.1, True)
        m_head = [
            ConvBlock(n_ch_in, n_feat, kernel_size, stride=2, bn=bn, bias=not bn, act=act)
        ]
        m_body = [
            ConvBlock(n_feat * (2 ** i), 2 * n_feat * (2 ** i), kernel_size, stride=2, bn=bn, bias=not bn, act=act)
            for i in range(n_conv)
        ]
        m_tail = [
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBlock(n_feat * (2 ** n_conv), n_ch_out, kernel_size=1, bias=True, bn=False, act=None),
        ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.norm = nn.Softmax(1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        out_w = self.tail(x)
        out_w = self.norm(out_w)
        return out_w


class ConvBlock(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            bn=False,
            bias=True,
            act=None,
    ):

        m = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, padding=(kernel_size // 2),
                stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(ConvBlock, self).__init__(*m)
