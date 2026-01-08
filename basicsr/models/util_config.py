from collections import OrderedDict
from typing import Dict, Any, Tuple

from basicsr.utils.logger import get_root_logger
from basicsr.utils.module_util import lookup_optimizer



def read_optimizer_options(train_opt, network, logger, is_discriminator=False):
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

    if partitioned_optimizer_opt is not None and hasattr(network, 'partitioned_parameters'):
        # Optimizer with multiple parameters group
        optimizer = _read_partitioned_optimizer(network, partitioned_optimizer_opt, logger)
    elif legacy_optimizer_opt is not None:
        # Fallback to global optimizer
        optimizer = _read_legacy_optimizer(network, legacy_optimizer_opt, logger)
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