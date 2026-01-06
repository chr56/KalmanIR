from typing import List

import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MultiBranchEDSR(nn.Module):
    """
    Multi-branch version of EDSR from paper "Enhanced Deep Residual Networks for Single Image Super-Resolution"
    Modified from https://github.com/sanghyun-son/EDSR-PyTorch
    """

    def __init__(
            self,
            branch=3,
            upscale=4,
            channel=3,
            img_range=1.,
            norm_image=False,
            n_blocks=16,
            n_features=64,
            res_scale=0.1,
    ):
        from basicsr.archs.components_EDSR import MeanShift, UpSampler, ResBlock
        super(MultiBranchEDSR, self).__init__()
        self.mean_norm = norm_image
        self.branch = branch
        self.upscale = upscale
        self.channel = channel

        kernel_size = 3
        padding = kernel_size // 2

        if self.mean_norm:
            self.sub_mean = MeanShift(img_range)
            self.add_mean = MeanShift(img_range, sign=1)

        # define head module
        m_head: List[nn.Module] = [nn.Conv2d(channel, n_features, kernel_size, padding=padding)]

        # define body module
        m_body: List[nn.Module] = [
            ResBlock(
                n_features, kernel_size, act=nn.ReLU(True), res_scale=res_scale
            ) for _ in range(n_blocks)
        ]
        m_body.append(nn.Conv2d(n_features, n_features, kernel_size, padding=padding))

        # define tail module
        if upscale == 1:
            m_tail: List[nn.Module] = [
                nn.Conv2d(n_features, channel * branch, kernel_size, padding=padding)
            ]
        else:
            m_tail: List[nn.Module] = [
                UpSampler(upscale, n_features, act=False),
                nn.Conv2d(n_features, channel * branch, kernel_size, padding=padding)
            ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        if self.mean_norm:
            x = self.sub_mean(x)

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        if self.mean_norm:
            x_branches = torch.split(x, self.channel, dim=1)
            processed_branches = [self.add_mean(b) for b in x_branches]
            x = torch.cat(processed_branches, dim=1)

        return x
