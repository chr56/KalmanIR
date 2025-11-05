from os import path as osp

from basicsr.utils.dynamic_registry import scan_and_import_modules, dynamic_instantiation
from basicsr.utils.registry import Registry

REFINEMENT_ARCH_REGISTRY = Registry('refinement_arch')

_current_dir = osp.dirname(osp.abspath(__file__))
_module_path = 'basicsr.archs.refinement'
_suffix = 'refinement_arch'

_modules = scan_and_import_modules(_current_dir, _module_path, _suffix)


def build_refinement_network(opt: dict, logging: bool = True):
    from copy import deepcopy
    from basicsr import get_root_logger

    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = REFINEMENT_ARCH_REGISTRY.get(network_type)(**opt)

    if logging:
        logger = get_root_logger()
        logger.info(f'Refinement Network [{net.__class__.__name__}] is created.')

    return net
