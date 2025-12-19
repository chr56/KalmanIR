import torch
from torch import nn
from torch.nn import functional as F

from . import MODULES_REGISTRY


@MODULES_REGISTRY.register()
class EnhancedKalmanPredictorMambaBlockV0(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            hidden_channel_expand: int = 24,
            num_layer: int = 2,
            final_activation: str = 'tanh',
            mamba_d_state: int = 8,
            mamba_d_expand: float = 2,
            mamba_dt_rank: int = 'auto',
            cab_compress_factor: int = 2,
            cab_squeeze_factor: int = 4,
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        from .residual_conv_block import ResidualConvBlock
        self.dim_expand = self.channels * hidden_channel_expand
        self.conv_block_expand = ResidualConvBlock(
            self.channels * 2, self.dim_expand, activation_type='leaky_relu'
        )
        self.conv_shrink = nn.Conv2d(
            self.dim_expand, self.channels, kernel_size=3, padding='same'
        )
        if final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_activation == 'none':
            self.final_activation = nn.Identity()
        else:
            raise NotImplementedError(f'Activation {final_activation} not implemented')

        from .modules_ss2d import MambaVSSBlock
        compressed_channels = self.dim_expand // cab_compress_factor
        squeezed_channels = self.channels // cab_squeeze_factor

        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(
                MambaVSSBlock(
                    dim=self.dim_expand,
                    mamba_d_state=mamba_d_state,
                    mamba_d_expand=mamba_d_expand,
                    mamba_dt_rank=mamba_dt_rank,
                    conv_d_compressed=compressed_channels,
                    conv_d_squeezed=squeezed_channels,
                    conv_leaky_relu=1e-2,
                )
            )

    def forward(self, image: torch.Tensor, difficult_zone: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, difficult_zone], dim=1)  # [B, C, H, W]

        x = self.conv_block_expand(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C']

        for layer in self.layers:
            x = layer(x)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.final_activation(self.conv_shrink(x))

        # [B, C, H, W]
        return x

    @staticmethod
    def model_input_format():
        return ['image', 'difficult_zone']


@MODULES_REGISTRY.register()
class EnhancedKalmanPredictorMambaBlockV1a(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            hidden_channel_expand: int = 24,
            num_layer: int = 2,
            final_activation: str = 'tanh',
            mamba_d_state: int = 8,
            mamba_d_expand: float = 2,
            mamba_dt_rank: int = 'auto',
            cab_compress_factor: int = 2,
            cab_squeeze_factor: int = 4,
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        from .residual_conv_block import ResidualConvBlock
        self.dim_expand = self.channels * hidden_channel_expand
        self.conv_block_expand_img = ResidualConvBlock(
            self.channels * 2, self.dim_expand,
            activation_type='tanh'
        )
        self.conv_block_expand_dz = ResidualConvBlock(
            self.channels, self.dim_expand,
            activation_type='leaky_relu', norm_type='instance'
        )
        self.conv_shrink = nn.Conv2d(
            self.dim_expand, self.channels, kernel_size=3, padding='same'
        )
        if final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_activation == 'none':
            self.final_activation = nn.Identity()
        else:
            raise NotImplementedError(f'Activation {final_activation} not implemented')

        from .modules_mamba_block import GuidedMambaVSSBlockV1
        compressed_channels = self.dim_expand // cab_compress_factor
        squeezed_channels = self.channels // cab_squeeze_factor

        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(
                GuidedMambaVSSBlockV1(
                    dim=self.dim_expand,
                    mamba_d_state=mamba_d_state,
                    mamba_d_expand=mamba_d_expand,
                    mamba_dt_rank=mamba_dt_rank,
                    bnc_compressed_channel=compressed_channels,
                    cca_squeezed_channel=squeezed_channels,
                    cca_leaky_relu=1e-2,
                )
            )

    def forward(self, image: torch.Tensor, difficult_zone: torch.Tensor) -> torch.Tensor:
        # image; [B, C, H, W]
        # difficult_zone: [B, C, H, W]

        x = self.conv_block_expand_img(torch.cat([image, difficult_zone.detach()], dim=1))  # [B, C', H, W]
        y = self.conv_block_expand_dz(difficult_zone)  # [B, C', H, W]

        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C']
        y = y.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C']

        for layer in self.layers:
            x = layer(x, y)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.final_activation(self.conv_shrink(x))

        # [B, C, H, W]
        return x

    @staticmethod
    def model_input_format():
        return ['image', 'difficult_zone']
