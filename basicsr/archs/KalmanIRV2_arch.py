# Code Implementation of the KalmanIR Model
from typing import Optional, Callable, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import init_weights
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils import decimal_to_binary, binary_to_decimal

from .IRNet import IRNet
from .modules_mamba import SS2DChanelFirst

from .kalman.kalman_refine_network_v2 import KalmanRefineNetV2


@ARCH_REGISTRY.register()
class KalmanIRV2(nn.Module):
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
                 d_state=16,
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
        super(KalmanIRV2, self).__init__()
        # define IR model
        out_chans = in_chans * 8
        self.ir_net = IRNet(
            img_size, patch_size, in_chans, out_chans, embed_dim, depths, drop_rate,
            d_state, mlp_ratio, drop_path_rate, norm_layer, patch_norm, use_checkpoint,
            upscale, img_range, upsampler, resi_connection, **kwargs
        )

        self.mamba_sigma = SS2DChanelFirst(d_model=out_chans)
        self.mamba_bias = SS2DChanelFirst(d_model=out_chans)

        self.kalman_refine = KalmanRefineNetV2(dim=out_chans)

        self.apply(init_weights)

    def flops(self):
        flops = 0
        flops += self.ir_net.flops()
        return flops

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

    def forward(self, x):

        x = self.ir_net(x)

        ##########################################
        # IR-1 Stage 粗步重建
        # shape of x: [B, 48, H, W]
        ##########################################

        binary_a = torch.sin(x[:, :24, ...])
        binary_b = torch.sin(x[:, 24:, ...])

        ##########################################
        # 误差估计
        ##########################################

        binary_refining = None
        sigma_t = None
        for i in range(1):
            if i == 0:
                sigma = self.cal_kl(binary_a, binary_b)
                sigma_t = self.mamba_sigma(sigma)
                bias = self.mamba_bias(binary_a)
                binary_refining = (binary_a / torch.exp(-sigma_t)) + bias
            else:
                sigma = self.cal_kl(binary_a, binary_refining)
                sigma_t = self.mamba_sigma(sigma)
                bias = self.mamba_bias(binary_a)
                binary_refining = (binary_a / torch.exp(-sigma_t)) + bias

        # binary_refining = binary_refining / self.img_range + self.mean
        binary_refined = torch.sin(binary_refining)

        ##########################################
        # Kalman Filter
        ##########################################

        binary_refined = self.kalman_refine(
            binary_refined, binary_a, binary_b, sigma_t,
        )

        ##########################################

        decimal_refined = binary_to_decimal(binary_refined)
        decimal_a = binary_to_decimal(binary_a)

        return [decimal_refined, decimal_a, binary_b]

