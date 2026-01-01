from typing import List

import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    """
    EDSR from paper "Enhanced Deep Residual Networks for Single Image Super-Resolution"
    Modified from https://github.com/sanghyun-son/EDSR-PyTorch
    """
    def __init__(
            self,
            upscale=4,
            channel=3,
            img_range=1.,
            n_blocks=16,
            n_features=64,
            res_scale=0.1,
    ):
        from basicsr.archs.components_EDSR import MeanShift, UpSampler, ResBlock
        super(EDSR, self).__init__()
        self.upscale = upscale
        self.channel = channel

        kernel_size = 3
        padding = kernel_size // 2

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
        m_tail: List[nn.Module] = [
            UpSampler(upscale, n_features, act=False),
            nn.Conv2d(n_features, channel, kernel_size, padding=padding)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
