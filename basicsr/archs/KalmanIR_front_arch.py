# Code Implementation of the KalmanIR Model
from typing import Optional, Callable, Tuple, Dict, Iterator
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
class KalmanIRFront(nn.Module):
    def __init__(self, in_chans=3, **kwargs):
        super(KalmanIRFront, self).__init__()
        out_chans = in_chans * 8
        self.ir_net = IRNet(in_chanel=in_chans, out_chanel=out_chans, **kwargs)

        self.mamba_error_estimate = MambaErrorEstimation(channel=out_chans)

        self.apply(init_weights)

    def partitioned_parameters(self)-> Dict[str, Iterator[nn.Parameter]]:
        from basicsr.utils.module_util import retrieve_parameters
        return {
            "backbone": retrieve_parameters(self.ir_net),
            "error_estimation": retrieve_parameters(self.mamba_error_estimate),
        }

    def flops(self):
        flops = 0
        flops += self.ir_net.flops()
        return flops

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

        binary_refined, sigma = self.mamba_error_estimate(binary_a, binary_b)
        binary_refined = torch.sin(binary_refined)

        ##########################################

        return [binary_refined, binary_a, binary_b]


class MambaErrorEstimation(nn.Module):
    def __init__(self, channel: int, **kwargs):
        super(MambaErrorEstimation, self).__init__()
        self.mamba_sigma = SS2DChanelFirst(d_model=channel, **kwargs)
        self.mamba_bias = SS2DChanelFirst(d_model=channel, **kwargs)

    def _cal_kl(self, pred, target):
        p = torch.log_softmax(pred, dim=1)
        q = torch.softmax(target, dim=1)
        return F.kl_div(p, q, reduction='none')

    def _forward_one_step(self, base, ref):
        kl = self._cal_kl(base, ref)
        sigma = self.mamba_sigma(kl)
        bias = self.mamba_bias(base)
        refined = (base / torch.exp(-sigma)) + bias
        return refined, sigma, bias

    def forward(self, a, b, iteration: int = 1):
        sigma = None
        refined = None
        for i in range(iteration):
            if i == 0:
                refined, sigma, _ = self._forward_one_step(base=a, ref=b)
            else:
                refined, sigma, _ = self._forward_one_step(base=a, ref=refined)

        return refined, sigma
