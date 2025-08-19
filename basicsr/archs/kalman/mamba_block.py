from typing import Optional, Type

import torch
from torch import nn
from timm.layers import DropPath


class VSSBlockFabric(nn.Module):
    """Basic block of VSSM, """
    def __init__(
            self,
            dim: int,
            ssm_block: Optional[nn.Module],
            mlp_block: Optional[nn.Module],
            post_norm: bool = False,
            norm_layer_type: Type[nn.Module] = nn.LayerNorm,
            residual_rate_ssm: Optional[float] = None,
            residual_rate_mlp: Optional[float] = None,
            drop_path_rate: float = 0.,
            use_checkpoint: bool = False,
    ):
        super(VSSBlockFabric, self).__init__()

        self.ssm = ssm_block
        self.mlp = mlp_block
        self.with_ssm_branch = ssm_block is not None
        self.with_mlp_branch = mlp_block is not None

        self.use_checkpoint = use_checkpoint
        self.drop_path = DropPath(drop_path_rate)
        self.post_norm = post_norm

        if self.with_ssm_branch:
            self.norm_ssm = norm_layer_type(dim)
            if residual_rate_ssm is None:
                self.residual_rate_ssm = nn.Parameter(torch.ones(dim))
            else:
                self.residual_rate_ssm = residual_rate_ssm

        if self.with_mlp_branch:
            self.norm_mlp = norm_layer_type(dim)
            if residual_rate_mlp is None:
                self.residual_rate_mlp = nn.Parameter(torch.ones(dim))
            else:
                self.residual_rate_mlp = residual_rate_mlp

    def ssm_forward(self, x):
        if self.post_norm:
            x = x * self.residual_rate_ssm + self.drop_path(self.norm_ssm(self.ssm(x)))
        else:
            x = x * self.residual_rate_ssm + self.drop_path(self.ssm(self.norm_ssm(x)))
        return x

    def mlp_forward(self, x):
        if self.post_norm:
            x = x * self.residual_rate_mlp + self.drop_path(self.norm_mlp(self.mlp(x)))
        else:
            x = x * self.residual_rate_mlp + self.drop_path(self.mlp(self.norm_mlp(x)))
        return x

    def all_forward(self, x: torch.Tensor):
        if self.with_ssm_branch:
            x = self.ssm_forward(x)
        if self.with_mlp_branch:
            x = self.mlp_forward(x)
        return x

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint:
            from torch.utils import checkpoint as tc
            return tc.checkpoint(self.all_forward, x)
        else:
            return self.all_forward(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# noinspection PyPep8Naming
class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=-1)
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x
