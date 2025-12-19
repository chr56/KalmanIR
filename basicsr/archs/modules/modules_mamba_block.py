import torch
from torch import nn
from torch.nn import functional as F


class GuidedMambaVSSBlockV1(nn.Module):
    def __init__(
            self,
            dim: int,
            mamba_d_state: int = 8,
            mamba_d_expand: float = 2.,
            mamba_dt_rank="auto",
            mamba_drop_path: float = 0,
            mamba_kwargs: dict = None,
            bnc_compressed_channel: int = 16,
            cca_squeezed_channel: int = 16,
            cca_leaky_relu: float = 0,
    ):
        if mamba_kwargs is None:
            mamba_kwargs = {}

        super().__init__()

        from basicsr.archs.modules.modules_ss2d import SS2D
        self.ss2d = SS2D(
            d_model=dim,
            d_state=mamba_d_state,
            dt_rank=mamba_dt_rank,
            expand=mamba_d_expand,
            dropout=mamba_drop_path,
            **mamba_kwargs
        )
        self.norm_ss2d = nn.LayerNorm(dim)
        self.skip_weight_ss2d = nn.Parameter(torch.ones(dim))

        from basicsr.archs.modules.channel_attention import ChannelCrossAttention
        self.cca = ChannelCrossAttention(
            dim, squeezed_channel=cca_squeezed_channel, leaky_relu=cca_leaky_relu, channel_last=True,
        )
        self.norm_cca = nn.LayerNorm(dim)
        self.skip_weight_cca = nn.Parameter(torch.ones(dim))
        from basicsr.archs.modules.channel_attention import BottleneckConv
        self.bnc = BottleneckConv(
            dim, compressed_channel=bnc_compressed_channel, channel_last=True,
        )
        self.norm_bnc = nn.LayerNorm(dim)
        self.skip_weight_bnc = nn.Parameter(torch.ones(dim))

    def forward_ss2d(self, x):
        residual = self.skip_weight_ss2d * x
        main = self.ss2d(self.norm_ss2d(x))
        return main + residual

    def forward_cca(self, x, dz):
        residual = self.skip_weight_cca * x
        main = self.cca(self.norm_cca(x), dz)
        return main + residual

    def forward_conv(self, x):
        residual = self.skip_weight_bnc * x
        main = self.bnc(self.norm_bnc(x))
        return main + residual

    def forward(self, x, difficult_zone):
        # x: [B, H, W, C]
        # difficult_zone: [B, H, W, C]

        x = self.forward_cca(x, difficult_zone)
        x = self.forward_ss2d(x)

        x = self.forward_conv(x)

        return x
