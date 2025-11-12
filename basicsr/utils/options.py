import argparse
import os
import random
from typing import List, Optional

import torch
import yaml
from collections import OrderedDict
from os import path as osp

from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt',
        type=str, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument(
        '--auto_resume',
        action='store_true')
    parser.add_argument(
        '--debug',
        action='store_true')
    parser.add_argument(
        '--local-rank',
        type=int, default=0)  # for pytorch 2.0
    parser.add_argument(
        '--force_yml',
        nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()
    return args


def parse_options(root_path, is_train=True):
    arguments = parse_arguments()
    with open(arguments.opt, mode='r') as f:
        # parse yml to dict
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    return process_options(
        opt=opt,
        launcher=arguments.launcher,
        auto_resume=arguments.auto_resume,
        debug=arguments.debug,
        force_yml=arguments.force_yml,
        root_path=root_path,
        is_train=is_train,
    ), arguments


def process_options(
        opt: OrderedDict,
        launcher: str,
        auto_resume: bool,
        debug: bool,
        force_yml: list,
        root_path: str,
        is_train=True,
):

    # distributed settings
    if launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if launcher == 'slurm' and 'dist_params' in opt:
            init_dist(launcher, **opt['dist_params'])
        else:
            init_dist(launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    # force to update yml options
    if force_yml is not None:
        for entry in force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    opt['auto_resume'] = auto_resume
    opt['is_train'] = is_train

    # debug setting
    if debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        # if dataset.get('dataroot_gt') is not None:
        #     dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        # if dataset.get('dataroot_lq') is not None:
        #     dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if is_train:
        experiments_root = osp.join(root_path, 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 20
            opt['logger']['print_freq'] = 4
            opt['logger']['save_checkpoint_freq'] = 40
    else:  # test
        results_root = osp.join(root_path, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt


@master_only
def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)


@master_only
def dump_option(opt: dict, path=None):
    try:
        path = path or osp.join(opt['path']['experiments_root'], opt['name']) + '.yml'
        os.makedirs(osp.dirname(path), exist_ok=True)
        with open(path, mode='w') as f:
            yaml.dump(opt, f, Dumper=ordered_yaml()[1])
    except Exception as e:
        print(f'Failed to dump option, {e}!')
        pass


class ValidationProfile:
    def __init__(self,
                 name: str,
                 iter_start: int,
                 iter_frequency: int,
                 datasets: List[str],
                 ):
        self.name = name
        self.start = iter_start
        self.frequency = iter_frequency
        self.datasets = datasets

    def check(self, current_iteration) -> bool:
        if current_iteration > 0 and self.frequency > 0:
            return current_iteration > self.start and (current_iteration - self.start) % self.frequency == 0
        else:
            return False

    def filter_datasets(self, val_loaders: list) -> list:

        if len(self.datasets) == 0:
            return val_loaders

        target_val_loaders = []
        for loader in val_loaders:
            dataset = loader.dataset
            if hasattr(dataset, 'opt'):
                dataset_name = dataset.opt.get('name', None)
                if dataset_name is not None:
                    if dataset_name in self.datasets:
                        target_val_loaders.append(loader)
                    continue
                else:
                    import warnings
                    warnings.warn(f"Dataset {dataset} is unnamed!")
            else:
                import warnings
                warnings.warn(f"Dataset {dataset} is invalid!")
        return target_val_loaders

    def __repr__(self):
        return (f"ValidationProfile({self.name}, "
                f"start={self.start}, frequency={self.frequency}), "
                f"datasets={self.datasets})")


def build_validation_profile(name: str, opt: dict, default: dict, debug: bool) -> ValidationProfile:
    start = int(opt.get('start', default.get('start', 0)))
    frequency = int(opt.get('frequency', default.get('frequency', default.get('val_freq', 0))))
    datasets = opt.get('datasets', [])
    if debug:
        start = int(start / 500)
        frequency = max(50, int(frequency / 500))
    return ValidationProfile(name, start, frequency, datasets)


def parse_val_profiles(val_opt: dict, debug: bool):
    if val_opt is None:
        return None

    profiles = OrderedDict()
    with_profile = False

    # look up for profiles
    for key, value in val_opt.items():
        if key.startswith('profile_'):
            with_profile = True
            name = key.split('_')[1]
            profiles[name] = build_validation_profile(name, value, val_opt, debug)

    if not with_profile:
        # no profile found, use globally
        profiles['default'] = build_validation_profile('default', val_opt, val_opt, debug)

    # profile that at end of training
    profile_end = val_opt.get('end')
    if profile_end is not None:
        profiles['end'] = build_validation_profile('end', profile_end, val_opt, debug)

    return profiles
