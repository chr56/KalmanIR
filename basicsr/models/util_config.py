from collections import OrderedDict
from typing import Dict, Any, Tuple

from basicsr.losses import build_loss
from basicsr.utils.logger import get_root_logger


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


def _extract(name, opt_, key, default):
    if key not in opt_:
        get_root_logger().warning(f"`{key}` is not defined in loss `{name}`, use default value `{default}`")
    value = opt_.get(key, default)
    return value
