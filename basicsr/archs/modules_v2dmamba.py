from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from timm.models.layers import DropPath

from .modules_common_ir import PatchUnEmbed, PatchEmbed
from .modules_channel_attention import CAB
from .vmamba_2d.vmamba import SS2D as VSS2D
from .vmamba_2d.vmamba import VSSM


##########################################

# using vmamba_2d
class BackboneVSSM(VSSM):
    def __init__(self, norm_layer: nn.Module, channel_first=True, **kwargs):
        super().__init__(**kwargs)
        del self.classifier
        self.channel_first = channel_first
        for i, dim in enumerate(self.dims):
            self.add_module(name=f'output_norm_layer_{i}', module=norm_layer(dim))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for i, layer in enumerate(self.layers):
            x = layer.blocks(x)
            norm_layer = getattr(self, f'output_norm_layer_{i}')
            x = norm_layer(x)
        if self.channel_first:
            x = x.permute(0, 3, 1, 2)
        return x


##########################################

# using vmamba_2d
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            feature_size: int,  # image_size // patch_size
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            dropout_rate: float = 0,
            d_state: int = 16,
            ssm_ratio: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ss2d = VSS2D(
            d_model=hidden_dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            act_layer=nn.SiLU,
            conv_bias=True,
            dropout=dropout_rate,
            bias=False,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            use_v2d=True,
            initialize="v2",
            forward_type="v05",
            channel_first=False,
            feature_size=feature_size,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.ss2d(x))
        x = (x * self.skip_scale2 +
             self.conv_blk(
                 self.ln_2(x).permute(0, 3, 1, 2).contiguous()
             ).permute(0, 2, 3, 1).contiguous()
             )
        x = x.view(B, -1, C).contiguous()
        return x


class BasicLayer(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 patch_size,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False, is_light_sr=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.feature_size = input_resolution[0] // patch_size

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                feature_size=self.feature_size,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                dropout_rate=0,
                d_state=d_state,
                ssm_ratio=self.mlp_ratio,
                input_resolution=input_resolution, is_light_sr=is_light_sr))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                from torch.utils import checkpoint
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 is_light_sr=False):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution  # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            patch_size=patch_size,
            depth=depth,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr=is_light_sr)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

##########################################

# using vmamba_2d
class VSS2ChanelFirst(VSS2D):
    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 3,
                 dropout_rate: float = 0,
                 expand_ratio: float = 2.,
                 channel_first: bool = True,
                 **kwargs):
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=expand_ratio,
            act_layer=nn.SiLU,
            d_conv=d_conv,
            dropout=dropout_rate,
            use_v2d=True,
            initialize="v2",
            forward_type="v05",
            channel_first=channel_first,
            feature_size=d_model,
            **kwargs
        )
        self.v2d_out_norm = None