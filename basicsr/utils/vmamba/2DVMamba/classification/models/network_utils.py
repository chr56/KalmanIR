import numbers
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import Parameter, init

import torch.nn.functional as F


class AffineFirstLayerNorm(nn.Module):
    __constants__ = ['affine_shape', 'normalized_shape', 'eps']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, affine_shape,  normalized_shape, eps: float = 1e-5,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]

        if isinstance(affine_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            affine_shape = (affine_shape,)  # type: ignore[assignment]
        self.affine_shape = tuple(affine_shape)  # type: ignore[arg-type]

        self.eps = eps

        self.weight = Parameter(torch.empty(self.affine_shape, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(self.affine_shape, **factory_kwargs))
        else:
            self.register_parameter('bias', None)


        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        input = input * self.weight + self.bias
        return F.layer_norm(
            input, self.normalized_shape, None, None, self.eps)

    def extra_repr(self) -> str:
        return '{affine_shape}, {normalized_shape}, eps={eps}, '.format(**self.__dict__)
