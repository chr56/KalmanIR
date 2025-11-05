from typing import Literal, Union, List, TypeVar

from torch import nn

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
        from .layer_norm import LayerNorm2d
        return LayerNorm2d(channel, eps=1e-6)
    elif norm_type == 'group':
        if norm_groups < 1: raise ValueError("norm_groups must be at least 1 for GroupNorm.")
        return nn.GroupNorm(num_groups=norm_groups, num_channels=channel, eps=1e-6)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm layer type: {norm_type}")


_T = TypeVar('_T')
OneOrMany = Union[_T, List[_T]]
