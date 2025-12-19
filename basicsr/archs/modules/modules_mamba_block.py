import torch
from torch import nn
from torch.nn import functional as F


class GuidedMambaVSSBlock(nn.Module):
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
            skip_weight_cca: float = 0.4,
            skip_weight_bnc: float = 1.0,
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
            dim, squeezed_channel=cca_squeezed_channel, leaky_relu=cca_leaky_relu,
        )
        self.norm_cca = nn.InstanceNorm2d(dim)
        self.skip_weight_cca = skip_weight_cca

        from basicsr.archs.modules.channel_attention import BottleneckConv
        self.bnc = BottleneckConv(
            dim, compressed_channel=bnc_compressed_channel
        )
        self.norm_bnc = nn.InstanceNorm2d(dim)
        self.skip_weight_bnc = skip_weight_bnc

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
        # x: [B, C, H, W]
        # difficult_zone: [B, C, H, W]

        x = self.forward_cca(x, difficult_zone)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.forward_ss2d(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.forward_conv(x)

        return x
