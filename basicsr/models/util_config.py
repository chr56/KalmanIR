from collections import OrderedDict
from typing import Dict, Any, Tuple

from basicsr.losses import build_loss
from basicsr.utils.logger import get_root_logger
from basicsr.utils.module_util import lookup_optimizer


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

    _supported_loss_mode = ['pixel', 'perceptual', 'gan']

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

    if len(criteria) == 0:
        raise ValueError('No losses defined.')
    logger.info(f"{len(criteria)} losses defined.")
    return criteria, require_discriminator


def read_optimizer_options(train_opt, net_g, logger):
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

    partitioned_optimizer_opt = train_opt.get('partitioned_optimizer_g', None)
    if partitioned_optimizer_opt is not None and hasattr(net_g, 'partitioned_parameters'):
        # Optimizer with multiple parameters group

        partitioned_parameters: Dict[str, Any] = net_g.partitioned_parameters()
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
                    f" ( available group: {list(partitioned_parameters.keys())} )"
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

        optimizer_g = lookup_optimizer(optim_type, params, **default_settings)
        logger.info(f"Created [{optim_type}] optimizer with {len(params)} parameters groups.")
    else:
        # Fallback to global optimizer
        optim_opt = train_opt['optim_g']

        params = []
        for k, v in net_g.named_parameters():
            if v.requires_grad:
                params.append(v)
            else:
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = optim_opt.pop('type')
        optimizer_g = lookup_optimizer(optim_type, params, **optim_opt)
        logger.info(f"Created global optimizer [{optim_type}].")

    return optimizer_g


def _extract(name, opt_, key, default):
    if key not in opt_:
        get_root_logger().warning(f"`{key}` is not defined in loss `{name}`, use default value `{default}`")
    value = opt_.get(key, default)
    return value
