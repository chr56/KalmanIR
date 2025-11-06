from typing import List

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


@MODULES_REGISTRY.register()
class KalmanPredictorMambaRecursiveV1a(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            iteration: int,
    ):
        super(KalmanPredictorMambaRecursiveV1a, self).__init__()

        from .layer_norm import LayerNorm2d
        from .utils_calculation import cal_kl
        from ..modules_mamba import SS2DChanelFirst

        self.channels = channels
        self.num_images = num_images
        self.iteration = iteration

        self.cal_kl = cal_kl
        self.norm_first = LayerNorm2d(channels)

        self.mamba_sigma = SS2DChanelFirst(d_model=channels)
        self.mamba_bias = SS2DChanelFirst(d_model=channels)

    def _forward_one_step(self, current_state, previous_state):
        kld = self.cal_kl(current_state, previous_state)
        sigma = self.mamba_sigma(kld)
        bias = self.mamba_bias(current_state)
        next_state = (current_state / torch.exp(-sigma)) + bias
        return next_state, current_state

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        previous = torch.zeros_like(image)
        current = self.norm_first(image)

        for i in range(self.iteration):
            current, previous = self._forward_one_step(current_state=current, previous_state=previous)

        current = torch.sin_(current)
        return current


@MODULES_REGISTRY.register()
class KalmanPredictorMambaRecursiveV1b(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            iteration: int,
    ):
        super(KalmanPredictorMambaRecursiveV1b, self).__init__()

        from .layer_norm import LayerNorm2d
        from .utils_calculation import cal_kl
        from ..modules_mamba import SS2DChanelFirst

        self.channels = channels
        self.num_images = num_images
        self.iteration = iteration

        self.cal_kl = cal_kl
        self.norm_first = LayerNorm2d(channels)

        self.mamba_sigma = SS2DChanelFirst(d_model=channels)
        self.mamba_bias = SS2DChanelFirst(d_model=channels)

    def _forward_one_step(self, current_state, previous_state):
        kld = self.cal_kl(current_state, previous_state)
        sigma = self.mamba_sigma(kld)
        bias = self.mamba_bias(current_state)
        next_state = (current_state / torch.exp(-sigma)) + bias
        return next_state, current_state

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        previous = torch.zeros_like(image)
        current = self.norm_first(image)

        for i in range(self.iteration):
            current, previous = self._forward_one_step(current_state=current, previous_state=previous.detach())

        current = torch.sin_(current)
        return current


@MODULES_REGISTRY.register()
class KalmanPredictorMambaRecursiveV1c(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            iteration: int,
    ):
        super(KalmanPredictorMambaRecursiveV1c, self).__init__()

        from .residual_conv_block import ResidualConvBlock
        from .layer_norm import LayerNorm2d
        from .utils_calculation import cal_kl
        from ..modules_mamba import SS2DChanelFirst

        self.conv_previous = ResidualConvBlock(
            in_channels=channels, out_channels=channels,
            activation_type='leaky_relu', norm_type='layer',
        )

        self.channels = channels
        self.num_images = num_images
        self.iteration = iteration

        self.cal_kl = cal_kl
        self.norm_first = LayerNorm2d(channels)

        self.mamba_sigma = SS2DChanelFirst(d_model=channels)
        self.mamba_bias = SS2DChanelFirst(d_model=channels)

    def _forward_one_step(self, current_state, previous_state):
        kld = self.cal_kl(current_state, previous_state)
        sigma = self.mamba_sigma(kld)
        bias = self.mamba_bias(current_state)
        next_state = (current_state / torch.exp(-sigma)) + bias
        return next_state, current_state

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        current = self.norm_first(image)
        previous = self.conv_previous(image).detach()

        for i in range(self.iteration):
            current, previous = self._forward_one_step(current_state=current, previous_state=previous.detach())

        current = torch.sin_(current)
        return current


@MODULES_REGISTRY.register()
class KalmanPredictorMambaRecursiveV2a(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            iteration: int,
    ):
        super(KalmanPredictorMambaRecursiveV2a, self).__init__()

        from .layer_norm import LayerNorm2d
        from .utils_calculation import cal_kl
        from ..modules_mamba import SS2DChanelFirst

        self.channels = channels
        self.num_images = num_images
        self.iteration = iteration

        self.cal_kl = cal_kl
        self.norm_first = LayerNorm2d(channels)

        self.mamba_sigma = SS2DChanelFirst(d_model=channels)
        self.mamba_bias = SS2DChanelFirst(d_model=channels)

    def _forward_one_step(self, current_state, previous_state):
        kld = self.cal_kl(current_state, previous_state)
        sigma = self.mamba_sigma(kld)
        bias = self.mamba_bias(current_state)
        next_state = (current_state / torch.exp(-sigma)) + bias
        return next_state, current_state

    def forward(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        previous = self.norm_first(previous)
        current = self.norm_first(current)

        for i in range(self.iteration):
            current, previous = self._forward_one_step(current_state=current, previous_state=previous)

        current = torch.sin_(current)
        return current


@MODULES_REGISTRY.register()
class KalmanPredictorMambaRecursiveV2b(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            iteration: int,
    ):
        super(KalmanPredictorMambaRecursiveV2b, self).__init__()

        from .layer_norm import LayerNorm2d
        from .utils_calculation import cal_kl
        from ..modules_mamba import SS2DChanelFirst

        self.channels = channels
        self.num_images = num_images
        self.iteration = iteration

        self.cal_kl = cal_kl
        self.norm_first = LayerNorm2d(channels)

        self.mamba_sigma = SS2DChanelFirst(d_model=channels)
        self.mamba_bias = SS2DChanelFirst(d_model=channels)

    def _forward_one_step(self, current_state, previous_state):
        kld = self.cal_kl(current_state, previous_state)
        sigma = self.mamba_sigma(kld)
        bias = self.mamba_bias(current_state)
        next_state = (current_state / torch.exp(-sigma)) + bias
        return next_state, current_state

    def forward(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        previous = self.norm_first(previous)
        current = self.norm_first(current)

        for i in range(self.iteration):
            current, previous = self._forward_one_step(current_state=current, previous_state=previous.detach())

        current = torch.sin_(current)
        return current


@MODULES_REGISTRY.register()
class KalmanPredictorMambaRecursiveV2c(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            iteration: int,
    ):
        super(KalmanPredictorMambaRecursiveV2c, self).__init__()

        from .residual_conv_block import ResidualConvBlock
        from .layer_norm import LayerNorm2d
        from .utils_calculation import cal_kl
        from ..modules_mamba import SS2DChanelFirst

        self.conv_first = ResidualConvBlock(
            in_channels=channels, out_channels=channels,
            activation_type='leaky_relu', norm_type='layer',
        )

        self.channels = channels
        self.num_images = num_images
        self.iteration = iteration

        self.cal_kl = cal_kl
        self.norm_first = LayerNorm2d(channels)

        self.mamba_sigma = SS2DChanelFirst(d_model=channels)
        self.mamba_bias = SS2DChanelFirst(d_model=channels)

    def _forward_one_step(self, current_state, previous_state):
        kld = self.cal_kl(current_state, previous_state)
        sigma = self.mamba_sigma(kld)
        bias = self.mamba_bias(current_state)
        next_state = (current_state / torch.exp(-sigma)) + bias
        return next_state, current_state

    def forward(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        current = self.conv_first(current)
        previous = self.conv_first(previous)

        for i in range(self.iteration):
            current, previous = self._forward_one_step(current_state=current, previous_state=previous.detach())

        current = torch.sin_(current)
        return current