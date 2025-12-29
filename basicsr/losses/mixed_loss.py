from dataclasses import dataclass
from typing import Dict, Any, Union, List, Callable

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

SUPPORTED_LOSS_MODE = ['pixel', 'perceptual', 'gan', 'gan_frozen', 'custom']

_DEFAULT_MODE = 'pixel'


@LOSS_REGISTRY.register()
class MixedLoss(nn.Module):
    def __init__(self, components, log_gan_output: bool = True):
        super(MixedLoss, self).__init__()
        if isinstance(components, dict):
            self.loss_components = nn.ModuleList([_create_loss_component_from_opt(opt) for name, opt in components.items()])
        elif isinstance(components, list):
            self.loss_components = nn.ModuleList([_create_loss_component_from_opt(opt) for opt in components])
        self.log_gan_output = log_gan_output

    def validate(self, available_output_names) -> bool:
        for loss in self.loss_components:
            if loss.target not in available_output_names:
                return False
        return True

    def summary(self) -> str:
        headers = ["Target", "Mode", "Type", "Weight", "Name"]
        col_widths = [18, 16, 18, 5, 10]

        separator = "-" * (sum(col_widths) + (len(col_widths) - 1) * 2)
        header_line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))

        lines = [separator, "Losses", separator, header_line, separator]

        for loss in self.loss_components:
            loss: _LossComponent
            loss_type = loss.loss_fn.__class__.__name__
            loss_weight = loss.loss_fn.loss_weight if hasattr(loss.loss_fn, 'loss_weight') else 'N/A'
            loss_name = loss.name if loss.name else ''
            line = "  ".join([
                f"{loss.target :<{col_widths[0]}}",
                f"{loss.mode :<{col_widths[1]}}",
                f"{loss_type :<{col_widths[2]}}",
                f"{loss_weight :<{col_widths[3]}}",
                f"{loss_name :<{col_widths[4]}}",
            ])
            lines.append(line)

        lines.append(separator)

        return "\n".join(lines)

    def forward(self, bundle: Dict[str, Any]):
        loss_total = 0
        loss_dict = dict()
        for loss_fn in self.loss_components:
            loss_total += loss_fn(bundle, loss_dict, self.log_gan_output)
        return loss_total, loss_dict


def _create_loss_component_from_opt(opt: dict) -> '_LossComponent':
    from . import build_loss
    loss_fn = build_loss(opt.get('loss'))

    target = opt.get('target', '*')

    name = opt.get('name') if 'name' in opt else None
    mode = str(opt.get('mode', _DEFAULT_MODE))
    assert mode in SUPPORTED_LOSS_MODE

    return _LossComponent(
        name=name,
        mode=mode,
        loss_fn=loss_fn,
        target=target,
    )


class _LossComponent(nn.Module):

    def __init__(self, name: Union[str, None], mode: str, target: str, loss_fn: nn.Module, transform=None):
        super(_LossComponent, self).__init__()
        self.name = name
        self.mode = mode
        self.target = target
        self.loss_fn = loss_fn
        self.transform = transform

    def forward(self, bundle: Dict[str, Any], loss_dict: dict, log_gan_output: bool = True):
        if self.name:
            loss_value_name = f'l_{self.target}_{self.name}'
        else:
            loss_value_name = f'l_{self.target}'

        if self.target != '*':
            prediction = bundle[self.target]
        else:
            prediction = bundle.values()

        if self.mode == 'pixel':
            loss_value = self.loss_fn(prediction, bundle['gt'])
            loss_dict[loss_value_name] = loss_value
        elif self.mode == 'gan':
            net_d = bundle['net_discriminator']
            d_out = net_d(prediction)
            loss_value = self.loss_fn(d_out, True)
            loss_dict[loss_value_name] = loss_value
            if log_gan_output:
                loss_dict[f'out_discr_{self.target}'] = torch.mean(d_out.detach())
        elif self.mode == 'gan_frozen':
            loss_value, d_out = self.loss_fn(prediction, True)
            loss_dict[loss_value_name] = loss_value
            if log_gan_output:
                loss_dict[f'out_discr_{self.target}'] = torch.mean(d_out.detach())
        elif self.mode == 'perceptual':
            loss_value = 0
            l_perceptual, l_style = self.loss_fn(prediction, bundle['gt'])
            if l_perceptual is not None:
                loss_value += l_perceptual
                loss_dict[loss_value_name if not l_style else f'{loss_value_name}_perceptual'] = l_perceptual
            if l_style is not None:
                loss_value += l_style
                loss_dict[loss_value_name if not l_perceptual else f'{loss_value_name}_style'] = l_style
        elif self.mode == 'custom':
            loss_value = self.loss_fn(bundle)
            loss_dict[loss_value_name] = loss_value
        else:
            raise NotImplementedError(f'mode {self.mode} is not supported')

        return loss_value
