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
class KalmanPredictorSimpleMambaV1(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            num_layer: int = 2,
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        self.num_layer = num_layer

        self.dim_expand = self.channels * 12
        self.linear_expand = nn.Conv2d(
            self.channels, self.dim_expand, kernel_size=3, padding='same'
        )
        self.linear_shrink = nn.Conv2d(
            self.dim_expand, self.channels, kernel_size=3, padding='same'
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(
                self._make_layer(self.dim_expand)
            )

    def _make_layer(self, channels: int) -> nn.Sequential:
        from .modules_ss2d import SS2DChanelFirst

        channels_group = channels // 6

        norm_ss2d = nn.GroupNorm(channels_group, channels)
        ss2d = SS2DChanelFirst(
            d_model=channels, d_state=8, dt_rank=channels_group
        )

        layer = nn.Sequential(
            norm_ss2d,
            ss2d,
        )
        return layer

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image
        x = self.linear_expand(x)
        for layer in self.layers:
            x = layer(x)
        x = self.linear_shrink(x)
        return x

    @staticmethod
    def model_input_format():
        return ['image']


@MODULES_REGISTRY.register()
class KalmanPredictorSimpleMambaV1f(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            num_layer: int = 2,
            with_linear_layers: bool = True,
            final_activation: str = 'sigmoid',
            hidden_channel_expand: int = 12,
            mamba_d_state: int = 8,
            mamba_d_rank: int = 4,
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        self.num_layer = num_layer

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

        self.mamba_d_state = mamba_d_state
        self.mamba_d_rank = mamba_d_rank
        self.with_linear_layers = with_linear_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(
                self._make_layer(self.dim_expand)
            )

    def _make_layer(self, channels: int) -> nn.Sequential:
        from .modules_ss2d import SS2DChanelFirst

        channels_group = channels // 6

        ss2d = SS2DChanelFirst(
            d_model=channels, d_state=self.mamba_d_state, dt_rank=self.mamba_d_rank
        )
        norm = nn.GroupNorm(channels_group, channels)

        if self.with_linear_layers:
            linear1 = nn.Conv2d(channels, channels, kernel_size=1, padding='same')
            act = nn.LeakyReLU(negative_slope=4e-2, inplace=True)
            linear2 = nn.Conv2d(channels, channels, kernel_size=1, padding='same')
            layer = nn.Sequential(
                ss2d,
                norm,
                linear1,
                act,
                linear2,
            )
        else:
            layer = nn.Sequential(
                ss2d,
                norm,
            )
        return layer

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image
        x = self.conv_block_expand(x)
        for layer in self.layers:
            x = layer(x)
        x = self.conv_shrink(x)
        x = self.final_activation(x)
        return x

    @staticmethod
    def model_input_format():
        return ['image']
