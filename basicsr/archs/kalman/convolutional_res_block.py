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
