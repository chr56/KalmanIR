from os import path as osp

from basicsr.utils.dynamic_registry import scan_and_import_modules
from basicsr.utils.registry import Registry

MODULES_REGISTRY = Registry('modules')

_current_dir = osp.dirname(osp.abspath(__file__))
_module_path = 'basicsr.archs.modules'
_suffix = 'module'

_modules = scan_and_import_modules(_current_dir, _module_path, _suffix)


def build_module(opt: dict, logging: bool = False):
    from copy import deepcopy
    from basicsr import get_root_logger

    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = MODULES_REGISTRY.get(network_type)(**opt)

    if logging:
        logger = get_root_logger()
        logger.info(f'Module [{net.__class__.__name__}] is created.')

    return net
