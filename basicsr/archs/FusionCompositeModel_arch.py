from typing import Dict, Iterator, Any, List

import torch
import torch.nn as nn

from basicsr.archs.arch_util import init_weights
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class FusionCompositeModel(nn.Module):

    def __init__(self,
                 upscale: int = 4,
                 channels: int = 3,
                 branch: int = 3,
                 branch_names: List[str] = None,
                 base_model: Dict[str, Any] = None,
                 refinement_model: Dict[str, Any] = None,
                 img_range: float = 1.0,
                 **kwargs):
        super(FusionCompositeModel, self).__init__()

        from basicsr.archs import build_network
        from basicsr.archs.refinement import build_refinement_network
        base_model.update(upscale=upscale, branch=branch, channel=channels, img_range=img_range)
        refinement_model.update(upscale=upscale, branch=branch, channels=channels, img_range=img_range)

        self.upscale = upscale

        self.channels = channels
        self.branch = branch
        self.branch_names = branch_names if branch_names is not None else [f"sr_{i + 1}" for i in range(branch)]
        assert len(self.branch_names) == self.branch, "`branch_names` size must be equal to `branch`"

        self.base_net: nn.Module = build_network(base_model)
        self.refinement_net: nn.Module = build_refinement_network(refinement_model)

        self.apply(init_weights)

    def partitioned_parameters(self) -> Dict[str, Iterator[nn.Parameter]]:
        from basicsr.utils.module_util import retrieve_parameters, retrieve_partitioned_parameters
        refinement_net_parameters = retrieve_partitioned_parameters(
            self.refinement_net, "refinement_net"
        )

        return {
            "base_model": retrieve_parameters(self.base_net),
            **refinement_net_parameters,
        }

    def forward(self, x):
        coarse_output = self.base_net(x)  # [B, n * C, H, W]

        coarse_branches = torch.chunk(coarse_output, self.branch, dim=1)  # n * [B, C, H, W]
        coarse_output = {
            self.branch_names[i]: coarse_branches[i]
            for i in range(self.branch)
        }  # dict { <branch_name> : [B, C, H, W] }

        refined_output: dict = self.refinement_net(coarse_branches)  # refined SR + other (e.g. Difficult Zone)

        return {**coarse_output, **refined_output}

    def model_output_format(self):
        format_base_model = {self.branch_names[i]: 'I' for i in range(self.branch)}
        format_refinement_net = self.refinement_net.model_output_format()

        return {
            **format_base_model,
            **format_refinement_net,
        }

    def primary_output(self):
        return self.refinement_net.primary_output()
