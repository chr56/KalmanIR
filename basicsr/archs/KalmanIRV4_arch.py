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

from .kalman.kalman_refine_network_v4 import KalmanRefineNetV4


@ARCH_REGISTRY.register()
class KalmanIRV4(nn.Module):
    """ KalmanIR Model
        :param in_chans(int): Number of input image channels. Default: 3
        :param kwargs: (the remaining arguments for backbone network)
       """

    def __init__(self,
                 in_chans=3,
                 uncertainty_estimation_mode='',
                 gain_calculation_mode='',
                 with_difficult_zone_affine=False,
                 **kwargs):
        super(KalmanIRV4, self).__init__()
        # define IR model
        out_chans = in_chans * 8
        self.ir_net = IRNet(in_chanel=in_chans, out_chanel=out_chans, **kwargs)

        self.mamba_sigma = SS2DChanelFirst(d_model=out_chans)
        self.mamba_bias = SS2DChanelFirst(d_model=out_chans)

        self.kalman_refine = KalmanRefineNetV4(
            dim=out_chans,
            img_seq=3,
            uncertainty_estimation_mode=uncertainty_estimation_mode,
            gain_calculation_mode=gain_calculation_mode,
            with_difficult_zone_affine=with_difficult_zone_affine,
        )

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
        binary_refined_1 = torch.sin(binary_refining)

        ##########################################
        # Kalman Filter
        ##########################################

        binary_refined_2 = self.kalman_refine(
            binary_refined_1, binary_a, binary_b, sigma_t,
        )

        ##########################################

        return [binary_refined_2, binary_refined_1, binary_a, binary_b]
