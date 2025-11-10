from typing import Dict, Iterator, Any

import torch
import torch.nn as nn

from basicsr.archs.arch_util import init_weights
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class CompositeModel(nn.Module):

    def __init__(self,
                 upscale: int = 4,
                 branch: int = 3,
                 channels: int = 3,
                 base_model_opt: Dict[str, Any] = None,
                 refinement_model_opt: Dict[str, Any] = None,
                 **kwargs):
        super(CompositeModel, self).__init__()

        from basicsr.archs.base import build_base_network
        from basicsr.archs.refinement import build_refinement_network
        base_model_opt.update(upscale=upscale, branch=branch, channels=channels)
        refinement_model_opt.update(upscale=upscale, branch=branch, channels=channels)

        self.upscale = upscale

        self.branch = branch
        self.channels = channels

        self.base_model: nn.Module = build_base_network(base_model_opt)
        self.refinement_net: nn.Module = build_refinement_network(refinement_model_opt)

        self.apply(init_weights)

    def partitioned_parameters(self) -> Dict[str, Iterator[nn.Parameter]]:
        from basicsr.utils.module_util import retrieve_parameters, retrieve_partitioned_parameters
        refinement_net_parameters = retrieve_partitioned_parameters(
            self.refinement_net, "refinement_net"
        )

        return {
            "base_model": retrieve_parameters(self.base_model),
            **refinement_net_parameters,
        }

    def forward(self, x):
        coarse_output = self.base_model(x)  # [B, n * C, H, W]

        coarse_branches = torch.chunk(coarse_output, self.branch, dim=1)  # n * [B, C, H, W]
        coarse_output = {
            f'sr_{i + 1}': coarse_branch
            for i, coarse_branch in enumerate(coarse_branches)
        } # dict { sr_{k} : [B, C, H, W] }

        refined_output: dict = self.refinement_net(coarse_branches) #  refined SR + other (e.g. Difficult Zone)

        return {**coarse_output, **refined_output}

    def model_output_format(self):
        format_base_model = {f'sr_{i + 1}': 'I' for i in range(self.branch)}
        format_refinement_net = self.refinement_net.model_output_format()

        return {
            **format_base_model,
            **format_refinement_net,
        }

    def primary_output(self):
        return self.refinement_net.primary_output()
