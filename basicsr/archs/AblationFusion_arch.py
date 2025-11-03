from typing import Dict, Iterator, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import init_weights
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class AblationFusion(nn.Module):

    def __init__(self,
                 upscale,
                 in_chans,
                 refinement_strategy: str,
                 refinement_arguments: Dict[str, Any],
                 ir_net_arguments: Dict[str, Any],
                 **kwargs):
        super(AblationFusion, self).__init__()

        out_chans = in_chans * 8

        from .IRNet import IRNet
        self.ir_net = IRNet(in_chanel=in_chans, out_chanel=out_chans, upscale=upscale, **ir_net_arguments)

        self.mamba_error_estimate = MambaErrorEstimation(channel=out_chans)

        if refinement_strategy == 'average':
            self.refinement_net = AverageFusion()
        elif refinement_strategy == 'learnable_weighted_addition':
            self.refinement_net = LearnableWeightedFusion(channels=out_chans, img_seq=3)
        elif refinement_strategy == 'kalman_v6':
            from .kalman.kalman_refine_network_v6 import KalmanRefineNetV6
            self.refinement_net = KalmanRefineNetV6(dim=out_chans, img_seq=3, **refinement_arguments)
        elif refinement_strategy == 'kalman_v4':
            from .kalman.kalman_refine_network_v4 import KalmanRefineNetV4
            self.refinement_net = KalmanRefineNetV4(dim=out_chans, img_seq=3, **refinement_arguments)
        else:
            raise NotImplementedError(f"Unknown refinement strategy {refinement_strategy}")

        self.apply(init_weights)

    def partitioned_parameters(self) -> Dict[str, Iterator[nn.Parameter]]:
        from basicsr.utils.module_util import retrieve_parameters, retrieve_partitioned_parameters
        refinement_net_parameters = retrieve_partitioned_parameters(self.refinement_net, "refinement")

        return {
            "backbone": retrieve_parameters(self.ir_net),
            "error_estimation": retrieve_parameters(self.mamba_error_estimate),
            **refinement_net_parameters,
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = self.ir_net(x)  # [B, 2 * 3 * 8, H, W]

        #########################################
        ############## DichotomyIR ##############
        #########################################

        binary_a = torch.sin(x[:, :24, ...])  # [B, 24, H, W]
        binary_b = torch.sin(x[:, 24:, ...])  # [B, 24, H, W]

        binary_d, sigma = self.mamba_error_estimate(binary_a, binary_b)
        binary_d = torch.sin(binary_d)

        #########################################
        ############## Refinement ###############
        #########################################

        binary_k = self.refinement_net(
            binary_d, binary_a, binary_b, sigma,
        )

        ##############################################

        return [binary_k, binary_d, binary_a, binary_b]


class MambaErrorEstimation(nn.Module):
    def __init__(self, channel: int, **kwargs):
        super(MambaErrorEstimation, self).__init__()
        from .modules_mamba import SS2DChanelFirst
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


class AverageFusion(nn.Module):
    def __init__(self, **kwargs):
        super(AverageFusion, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, sigma) -> torch.Tensor:
        return (a + b + c) / 3


class LearnableWeightedFusion(nn.Module):
    def __init__(self, channels: int, img_seq: int = 3, **kwargs):
        super(LearnableWeightedFusion, self).__init__()
        self.conv = nn.Conv2d(img_seq * channels, channels, kernel_size=1, stride=1, padding='same')

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, sigma) -> torch.Tensor:
        x = torch.cat([a, b, c], dim=1)
        x = self.conv(x)
        x = F.sigmoid(x)
        return x
