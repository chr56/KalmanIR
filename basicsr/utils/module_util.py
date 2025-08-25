from typing import Iterator, Dict

import torch
import torch.nn as nn

from basicsr.utils import get_root_logger


def retrieve_parameters(module: nn.Module, exclude_non_trainable: bool = True) -> Iterator[nn.Parameter]:
    logger = get_root_logger()
    for name, param in module.named_parameters():
        if param.requires_grad or not exclude_non_trainable:
            yield param
        else:
            logger.warning(f'Params {name} in {module.__class__.__name__} will not be included.')


def retrieve_partitioned_parameters(
        module: nn.Module,
        free_parameters_group_name: str,
        exclude_non_trainable: bool = True
) -> Dict[str, Iterator[nn.Parameter]]:
    parameter_groups = {}
    # submodules
    for submodule_name, submodule in module.named_children():
        parameter_groups[submodule_name] = retrieve_parameters(submodule, exclude_non_trainable=exclude_non_trainable)
    # current module
    parameter_groups[free_parameters_group_name] = module.parameters(recurse=False)
    return parameter_groups


def lookup_optimizer(optimizer_name: str, params, **kwargs) -> torch.optim.Optimizer:
    """
    :param optimizer_name: name of the optimizer
    :param params: modules parameters to be optimized by
    :param kwargs: optimizer options
    :return: torch Optimizer
    """
    optimizer_name = optimizer_name.lower()
    # noinspection SpellCheckingInspection
    if optimizer_name == 'sgd':
        return torch.optim.SGD(params, **kwargs)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(params, **kwargs)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(params, **kwargs)
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(params, **kwargs)
    elif optimizer_name == 'adagrad':
        return torch.optim.Adagrad(params, **kwargs)
    elif optimizer_name == 'adamax':
        return torch.optim.Adamax(params, **kwargs)
    elif optimizer_name == 'lbfgs':
        return torch.optim.LBFGS(params, **kwargs)
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not implemented.")
