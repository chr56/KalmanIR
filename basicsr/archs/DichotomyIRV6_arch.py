# Code Implementation of the MambaIR Model

'''
use 2DVMamba
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils import decimal_to_binary, binary_to_decimal
from timm.models.layers import trunc_normal_

NEG_INF = -1000000

from .modules_mamba import SS2DChanelFirst
from .modules_common_ir import Upsample, UpsampleOneStep

# from .modules_v2dmamba import BackboneVSSM

@ARCH_REGISTRY.register()
class DichotomyIRV6(nn.Module):
    r""" MambaIR Model
           A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.

       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
       """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state = 16,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(DichotomyIRV6, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans*8
        num_feat = 64
        self.img_size = img_size
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio=mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.embed_dim = embed_dim
        self.num_features = embed_dim

        from .modules_v2dmamba import BackboneVSSM

        layers_dim = [embed_dim for _ in range(len(depths))]
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.features_extractor = BackboneVSSM(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            depths=depths,
            dims=layers_dim,
            ssm_d_state=d_state,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ssm_init="v2", forward_type="v05", use_v2d=True,
        )

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(

                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # -------------------------3. high-quality image reconstruction ------------------------ #
        # self.downsample = nn.AvgPool2d(kernel_size=self.upscale, stride=self.upscale)
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, 2*num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, 2*num_out_ch)

        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, 2*num_out_ch, 3, 1, 1)

        self.mamba_g = SS2DChanelFirst(d_model=num_out_ch)
        self.mamba_b = SS2DChanelFirst(d_model=num_out_ch)
        # self.last_norm = nn.BatchNorm2d(num_features=24)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def cal_kl(self, pred, target):

        P = torch.log_softmax(pred, dim=1)
        Q = torch.softmax(target, dim=1)

        return F.kl_div(P, Q, reduction='none')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.pos_drop(x)
        x = self.features_extractor(x)
        return x

    def forward(self, x):
        # self.mean = self.mean.type_as(x)
        # x = (x - self.mean) * self.img_range
        # b, _, _, _ = x.shape
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            feat = self.forward_features(x)
            x = self.conv_after_body(feat) + x
            feat_before_upsample = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(feat_before_upsample))

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            feat = self.forward_features(x)
            x = self.conv_after_body(feat) + x
            x = self.upsample(x)

        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        b_L1 = torch.sin(x[:, :24, ...])
        b_BCE = torch.sin(x[:, 24:, ...])

        x_decimal_1 = binary_to_decimal(b_L1)#[bx24xhxw]
        x_binary_1 = torch.sigmoid(b_BCE)

        # sigma = self.cal_kl(b_L1, b_BCE)
        # sigma_t = self.mamba_g(sigma)
        # bias = self.mamba_b(b_L1)
        # x = (b_L1 / torch.exp(- sigma_t)) + bias
        for i in range(1):
            if i == 0:
                sigma = self.cal_kl(b_L1, b_BCE)
                sigma_t = self.mamba_g(sigma)
                bias = self.mamba_b(b_L1)
                x = (b_L1 / torch.exp(-sigma_t)) + bias
            else:
                sigma = self.cal_kl(b_L1, x)
                sigma_t = self.mamba_g(sigma)
                bias = self.mamba_b(b_L1)
                x = (b_L1 / torch.exp(-sigma_t)) + bias

        x_binary_2 = torch.sin(x)
        x_decimal_2 = binary_to_decimal(x_binary_2)
        # x = x / self.img_range + self.mean
        return [x_decimal_2, x_decimal_1, x_binary_1]

    def flops(self):
        flops = 0
        h, w = self.img_size, self.img_size
        flops += h * w * 3 * self.embed_dim * 9
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops
