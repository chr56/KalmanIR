from collections import OrderedDict
from functools import partial
from typing import Dict, Any, Tuple

from torch import Tensor

from basicsr.losses import build_loss
from basicsr.utils.binary_transform import decimal_to_binary, binary_to_decimal
from basicsr.utils.logger import get_root_logger
from basicsr.utils.module_util import lookup_optimizer


def config_suffix(is_discriminator: bool) -> str:
    return 'd' if is_discriminator else 'g'

def frozen_model_parameters(net, frozen: bool):
    for p in net.parameters():
        p.requires_grad = (not frozen)

def valid_model_output_settings(model_output_format, enabled_output_indexes):
    # outputs
    _supported_formats = ['B', 'D', 'I']
    if isinstance(model_output_format, list):
        for f in model_output_format:
            if f not in _supported_formats:
                raise NotImplementedError(f"Unsupported format: {model_output_format}")
    else:
        raise ValueError("`model_output` should be a list")
    # enabled outputs
    if enabled_output_indexes is None:
        enabled_output_indexes = [0]  # enable first output by default
    elif isinstance(enabled_output_indexes, list):
        for idx in enabled_output_indexes:
            assert isinstance(idx, int) and 0 <= idx < len(model_output_format), ValueError(
                f"illegal index `{idx}` in `model_output_enabled`: {model_output_format}"
            )
    else:
        raise ValueError("`model_output_enabled` should be a list")
    primary_output_index = enabled_output_indexes[0]  # first enabled output as primary output
    return primary_output_index


def read_loss_options(train_opt, device, output_length: int, logger) -> Tuple[dict, bool]:
    r"""
    Example of `train_opt`
    ```
    losses:
      <LossName1>:
        mode: 'pixel' # 'perceptual'
        target: 0
        format: 'D'
        loss:
          type: <LossType>
          loss_weight: 1.0
          reduction: mean
          ...
      <LossName2>:
        mode: 'pixel' # 'gan'
        target: 2
        format: 'B'
        loss:
          type: <LossType>
          loss_weight: 0.1
          reduction: mean
          ...
    ```
    """

    criteria = OrderedDict()

    _supported_loss_mode = ['pixel', 'perceptual', 'gan', 'vae']

    require_discriminator = False
    if train_opt.get('losses'):
        # multiple losses
        for name, loss_opt in train_opt['losses'].items():
            if 'loss' not in loss_opt:
                logger.error(f"Loss type `{name}` is not defined in loss `{name}`, skipping.")
                continue
            loss_fn = build_loss(loss_opt['loss']).to(device)

            loss_target = int(_extract(name, loss_opt, 'target', 0))
            loss_format = _extract(name, loss_opt, 'format', 'D')
            loss_mode = str(_extract(name, loss_opt, 'mode', 'pixel'))
            if loss_mode not in _supported_loss_mode:
                logger.error(f"unsupported loss mode `{loss_mode}` in loss `{name}`")
                loss_mode = _supported_loss_mode[0]
            if loss_target < 0 or loss_target >= output_length:
                logger.error(f"outraged loss target index `{loss_target}` in loss `{name}`")
                loss_target = 0

            if loss_mode == 'gan':
                require_discriminator = True

            criteria[name] = {
                'loss': loss_fn,
                'target': loss_target,
                'format': loss_format,
                'mode': loss_mode,
            }
    else:
        # simple mixed losses
        if train_opt.get('pixel_opt'):
            loss_fn = build_loss(train_opt['pixel_opt']).to(device)
            criteria['pixel'] = {
                'loss': loss_fn,
                'target': None,
                'format': 'D',
                'mode': 'pixel',
            }
        if train_opt.get('perceptual_opt'):
            loss_fn = build_loss(train_opt['perceptual_opt']).to(device)
            criteria['perceptual'] = {
                'loss': loss_fn,
                'target': None,
                'format': 'D',
                'mode': 'perceptual',
            }
        if train_opt.get('vae_opt'):
            loss_fn = build_loss(train_opt['vae_opt']).to(device)
            criteria['perceptual'] = {
                'loss': loss_fn,
                'target': None,
                'format': 'I',
                'mode': 'vae',
            }

    if len(criteria) == 0:
        raise ValueError('No losses defined.')
    logger.info(f"{len(criteria)} losses defined.")
    return criteria, require_discriminator


class MultipleLossOptions:

    def __init__(self, losses_options: dict, model_output_format, gt_format):
        self.losses_per_output: Dict[int, list] = dict()
        self.all_gan_losses = list()

        for name, loss in losses_options.items():
            self.register(name, loss, gt_format, model_output_format)
        pass

    def register(self, name, loss, gt_format, model_output_format):

        target_idx = loss['target']

        loss_mode = loss['mode']
        loss_fn = loss['loss']

        target_format = loss['format']

        if target_idx is None:
            # mixed loss
            origin_format = None
            output_transform = lambda outputs: outputs
        else:
            # specified loss
            origin_format = model_output_format[target_idx]
            output_transform = lambda outputs: convert_format(
                outputs[target_idx], from_format=origin_format, to_format=target_format
            )

        gt_transform = lambda gt: convert_format(gt, from_format=gt_format, to_format=target_format)

        item = {
            'name': name,
            'mode': loss_mode,
            'loss_fn': loss_fn,
            'target_idx': target_idx,
            'target_format': target_format,
            'origin_format': origin_format,
            'output_transform': output_transform,
            'gt_transform': gt_transform,
        }

        output_index = target_idx if target_idx is not None else -1
        if self.losses_per_output.get(output_index, None) is None:
            self.losses_per_output[output_index] = [item]
        else:
            self.losses_per_output[output_index].append(item)
        pass

        if loss_mode == 'gan':
            self.all_gan_losses.append(item)

    def losses(self):
        return self.losses_per_output

    def __repr__(self) -> str:
        headers = ["#", "Name", "Mode", "Type", "Format"]
        col_widths = [3, 12, 12, 15, 8]

        separator = "-" * (sum(col_widths) + (len(col_widths) - 1) * 2)
        header_line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))

        lines = [separator, "Losses", separator, header_line, separator]

        for output_index, losses in self.losses_per_output.items():
            for loss in losses:
                loss_type = loss['loss_fn'].__class__.__name__
                if output_index == -1:
                    converting = f"(Mixed)"
                    index_name = 'ALL'
                else:
                    converting = f"{loss['origin_format']} -> {loss['target_format']}"
                    index_name = str(output_index)
                line = "  ".join([
                    f"{index_name :<{col_widths[0]}}",
                    f"{loss['name']:<{col_widths[1]}}",
                    f"{loss['mode']:<{col_widths[2]}}",
                    f"{loss_type:<{col_widths[3]}}",
                    f"{converting:<{col_widths[4]}}",
                ])
                lines.append(line)

        lines.append(separator)

        return "\n".join(lines)


def convert_format(tensor: Tensor, from_format, to_format) -> Tensor:
    if from_format == to_format:
        return tensor
    elif from_format == 'D' and to_format == 'B':
        return decimal_to_binary((tensor * 255.))
    elif from_format == 'B' and to_format == 'D':
        return binary_to_decimal(tensor)
    elif to_format == 'I':
        return tensor # Identical
    else:
        raise NotImplementedError(f"`Conversion {from_format} -> {to_format}` is not implemented.")


def apply_parameter_frozen_settings(target_network, target_names: list, frozen: bool):
    logger = get_root_logger()
    grouped = target_network.partitioned_parameters()
    assert isinstance(target_names, list), f"frozen parameter names must be a list"

    for module_name in target_names:
        parameters = grouped[module_name]
        logger.info(f"Frozen module {module_name} parameters...")
        for p in parameters:
            p.requires_grad = (not frozen)
        pass


def read_optimizer_options(train_opt, net_g, logger, is_discriminator=False):
    """
    Example of `train_opt`
    ```
      partitioned_optimizer_g:
        type: Adam
        default:
          lr: !!float 3e-4
          weight_decay: 0
          betas: [ 0.9, 0.99 ]
        params:
          <GroupName1>:
            lr: !!float 5e-4
            weight_decay: !!float 1e-5
            betas: [ 0.9, 0.99 ]
          <GroupName2>:
            lr: !!float 6e-4
            weight_decay: !!float 1e-5
            betas: [ 0.9, 0.99 ]
    ```
    """

    suffix = 'g' if not is_discriminator else 'd'

    partitioned_optimizer_opt = train_opt.get(f"partitioned_optimizer_{suffix}", None)
    legacy_optimizer_opt = train_opt.get(f"optim_{suffix}", None)

    if partitioned_optimizer_opt is not None and hasattr(net_g, 'partitioned_parameters'):
        # Optimizer with multiple parameters group
        optimizer = _read_partitioned_optimizer(net_g, partitioned_optimizer_opt, logger)
    elif legacy_optimizer_opt is not None:
        # Fallback to global optimizer
        optimizer = _read_legacy_optimizer(net_g, legacy_optimizer_opt, logger)
    else:
        raise ValueError('Optimizer option not defined.')

    return optimizer


def _read_partitioned_optimizer(model, partitioned_optimizer_opt, logger):
    partitioned_parameters: Dict[str, Any] = model.partitioned_parameters()

    remained_parameters_groups = list(partitioned_parameters.keys())
    optim_type = partitioned_optimizer_opt['type']
    default_settings = partitioned_optimizer_opt['default']
    params = []
    parameters_groups = partitioned_optimizer_opt.get('params', None)
    if parameters_groups is not None:
        for group_name, group_settings in parameters_groups.items():
            params_group = partitioned_parameters.get(group_name, None)
            assert params_group is not None, ValueError(
                f"Parameters group '{group_name}' not found"
                f" (available group: {list(partitioned_parameters.keys())})"
            )
            params.append(
                {'params': params_group, **group_settings}
            )
            remained_parameters_groups.remove(group_name)
        pass
    if len(remained_parameters_groups) > 0:
        remaining_params_group = []
        for key in remained_parameters_groups:
            remaining_params_group.extend(partitioned_parameters[key])
        params.append(
            {'params': remaining_params_group, **default_settings}
        )
        pass
    optimizer = lookup_optimizer(optim_type, params, **default_settings)
    model_name = model.__class__.__name__
    logger.info(f"Created [{optim_type}] optimizer with {len(params)} parameters groups for [{model_name}].")
    return optimizer


def _read_legacy_optimizer(model, optim_opt, logger):
    params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            params.append(v)
        else:
            logger.warning(f"Params {k} will not be optimized.")
    optim_type = optim_opt.pop('type')
    optimizer = lookup_optimizer(optim_type, params, **optim_opt)
    model_name = model.__class__.__name__
    logger.info(f"Created a global optimizer [{optim_type}] for [{model_name}].")
    return optimizer


def _extract(name, opt_, key, default):
    if key not in opt_:
        get_root_logger().warning(f"`{key}` is not defined in loss `{name}`, use default value `{default}`")
    value = opt_.get(key, default)
    return value
