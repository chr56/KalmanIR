import torch
from torch import nn
from torch.nn import functional as F

from . import MODULES_REGISTRY


@MODULES_REGISTRY.register()
class KalmanPredictorV1(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
    ):
        super(KalmanPredictorV1, self).__init__()

        self.channels = channels
        self.num_images = num_images

        from .residual_conv_block import ResidualConvBlock
        self.conv_block_1 = ResidualConvBlock(
            in_channels=channels, out_channels=channels,
            activation_type='leaky_relu', norm_type='layer',
        )
        self.conv_block_2 = ResidualConvBlock(
            in_channels=channels, out_channels=channels,
            activation_type='leaky_relu', norm_type='layer',
        )
        self.conv_block_3 = ResidualConvBlock(
            in_channels=channels, out_channels=channels,
            activation_type='silu', norm_type='layer',
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        return x

    @staticmethod
    def model_input_format():
        return ['image']


@MODULES_REGISTRY.register()
class KalmanPredictorMambaBlockV2x(nn.Module):
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
            self.channels, self.dim_expand, activation_type='leaky_relu'
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image  # [B, C, H, W]

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
        return ['image']


from .utils import NormLayerType, ActivationFunction, get_activation_function


@MODULES_REGISTRY.register()
class KalmanPredictorMambaBlockV3base(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            hidden_channel_expand: float = 24,
            num_layer: int = 2,
            mamba_d_state: int = 8,
            mamba_d_expand: float = 2,
            mamba_dt_rank: int = 'auto',
            cab_compress_factor: int = 2,
            cab_squeeze_factor: int = 4,
            conv_expand_layer: int = 2,
            conv_expand_kernel_size: int = 3,
            conv_expand_norm: NormLayerType = 'instance',
            conv_expand_activation: ActivationFunction = 'leaky_relu',
            conv_shrink_kernel_size: int = 3,
            final_activation: ActivationFunction = 'tanh',
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        from .residual_conv_block import ResidualConvBlock
        self.dim_expand = int(self.channels * hidden_channel_expand)
        self.conv_block_expand = ResidualConvBlock(
            self.channels, self.dim_expand,
            num_layers=conv_expand_layer, kernel_size=conv_expand_kernel_size,
            activation_type=conv_expand_activation, norm_type=conv_expand_norm,
        )
        self.conv_shrink = nn.Conv2d(
            self.dim_expand, self.channels,
            kernel_size=conv_shrink_kernel_size, padding='same',
        )
        self.final_activation = get_activation_function(final_activation, required=False)

        from .modules_ss2d import MambaVSSBlock
        compressed_channels = self.dim_expand // cab_compress_factor
        squeezed_channels = self.dim_expand // cab_squeeze_factor

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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image  # [B, C, H, W]

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
        return ['image']
