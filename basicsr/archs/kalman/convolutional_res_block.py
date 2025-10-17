from typing import Callable, Literal, Union, List, TypeVar

import torch
from torch import nn

_T = TypeVar('_T')
OneOrMany = Union[_T, List[_T]]


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


ActivationFunction = Literal['relu', 'leaky_relu', 'silu', 'sigmoid']


def get_activation_function(activation_type, required: bool = True, **kwargs):
    if activation_type == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_type == 'leaky_relu':
        return nn.LeakyReLU(inplace=True, **kwargs)
    elif activation_type == 'silu':
        return nn.SiLU(inplace=True)
    elif activation_type == 'sigmoid':
        return nn.Sigmoid()
    else:
        if required:
            raise NotImplementedError(f"activation type `{activation_type}` is unimplemented")
        else:
            return nn.Identity()


NormLayerType = Literal['batch', 'layer', 'group', 'none']


def get_norm_layer(norm_type: NormLayerType, channel: int, norm_groups: int = -1) -> nn.Module:
    if norm_type == 'batch':
        return nn.BatchNorm2d(channel, eps=1e-6)
    elif norm_type == 'layer':
        from .utils import LayerNorm2d
        return LayerNorm2d(channel, eps=1e-6)
    elif norm_type == 'group':
        if norm_groups < 1: raise ValueError("norm_groups must be at least 1 for GroupNorm.")
        return nn.GroupNorm(num_groups=norm_groups, num_channels=channel, eps=1e-6)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm layer type: {norm_type}")


class ResidualConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int = -1,
            num_layers: int = 2,
            kernel_size: OneOrMany[int] = 3,
            norm_type: OneOrMany[NormLayerType] = 'layer',
            norm_group: OneOrMany[int] = 3,
            activation_type: OneOrMany[ActivationFunction] = 'relu',
            activation_before_conv: bool = False,
            norm_after_conv: bool = False
    ):
        super().__init__()

        out_channels = out_channels if out_channels > 0 else in_channels

        if isinstance(activation_type, str):
            activation_type = [activation_type] * num_layers
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * num_layers
        if isinstance(norm_type, str):
            norm_type = [norm_type] * num_layers
        if isinstance(norm_group, int):
            norm_group = [norm_group] * num_layers

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.activation_first = activation_before_conv
        self.post_norm = norm_after_conv

        self.norm_layers = nn.ModuleList()
        self.activations = nn.ModuleList([get_activation_function(t) for t in activation_type])
        self.conv_layers = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            out_ch = out_channels
            norm_ch = in_ch if i == 0 and not self.post_norm else out_ch

            conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size[i], padding='same')
            self.conv_layers.append(conv)

            norm = get_norm_layer(norm_type[i], norm_ch, norm_group[i])
            self.norm_layers.append(norm)

        if in_channels != out_channels:
            self.conv_residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')

    def _forward_layer(self, x, layer):
        if not self.post_norm:
            x = self.norm_layers[layer](x)
        x = self.conv_layers[layer](x)
        x = self.activations[layer](x)
        if self.post_norm:
            x = self.norm_layers[layer](x)
        return x

    def _forward_layer_activation_first(self, x, layer):
        if not self.post_norm:
            x = self.norm_layers[layer](x)
        x = self.activations[layer](x)
        x = self.conv_layers[layer](x)
        if self.post_norm:
            x = self.norm_layers[layer](x)
        return x

    def forward(self, x_in):
        x = x_in

        for i in range(self.num_layers):
            if self.activation_first:
                x = self._forward_layer_activation_first(x, layer=i)
            else:
                x = self._forward_layer(x, layer=i)

        if self.in_channels != self.out_channels:
            x_in = self.conv_residual(x_in)

        return x + x_in


class _BaseConvolutionalResBlock(nn.Module):

    def __init__(
            self,
            norm_layer1: nn.Module,
            norm_layer2: nn.Module,
            channels: int,
            out_channels: int = -1,
            activation_type: ActivationFunction = 'relu',
            kernel_size: int = 3
    ):
        super(_BaseConvolutionalResBlock, self).__init__()

        out_channels = out_channels if out_channels > 0 else channels

        self.channels = channels
        self.out_channels = out_channels

        self.norm1 = norm_layer1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding='same', stride=1)

        self.norm2 = norm_layer2
        self.conv2 = nn.Conv2d(channels, out_channels, kernel_size=kernel_size, padding='same', stride=1)

        if channels != out_channels:
            self.conv_residual = nn.Conv2d(channels, out_channels, kernel_size=1, padding='same')

        self.activation = get_activation_function(activation_type)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        if self.channels != self.out_channels:
            x_in = self.conv_residual(x_in)

        return x + x_in


class ConvolutionalResBlockLayerNorm(_BaseConvolutionalResBlock):

    def __init__(
            self,
            channels: int,
            out_channels: int = -1,
            activation_type: ActivationFunction = 'relu',
            kernel_size: int = 3
    ):
        from .utils import LayerNorm2d
        norm1 = LayerNorm2d(channels, eps=1e-6, elementwise_affine=True)
        norm2 = LayerNorm2d(channels, eps=1e-6, elementwise_affine=True)

        super(ConvolutionalResBlockLayerNorm, self).__init__(
            norm_layer1=norm1,
            norm_layer2=norm2,
            channels=channels,
            out_channels=out_channels,
            activation_type=activation_type,
            kernel_size=kernel_size
        )


class ConvolutionalResBlockGroupNorm(_BaseConvolutionalResBlock):

    def __init__(
            self,
            channels: int,
            norm_group: int = 1,
            out_channels: int = -1,
            activation_type: ActivationFunction = 'relu',
            kernel_size: int = 3
    ):
        norm1 = nn.GroupNorm(norm_group, channels, eps=1e-6)
        norm2 = nn.GroupNorm(norm_group, channels, eps=1e-6)

        super(ConvolutionalResBlockGroupNorm, self).__init__(
            norm_layer1=norm1,
            norm_layer2=norm2,
            channels=channels,
            out_channels=out_channels,
            activation_type=activation_type,
            kernel_size=kernel_size
        )
