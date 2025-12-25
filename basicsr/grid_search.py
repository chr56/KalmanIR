import sys
from collections import OrderedDict
from os import path as osp
from time import sleep
from typing import List

import yaml

ROOT_PATH = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(ROOT_PATH)  # Add the parent directory to sys.path for `basicsr`

from basicsr.utils.options import (
    parse_arguments, setup_env_and_update_options, ordered_yaml
)
from basicsr.train import train_pipeline


def _replace_placeholder(it, placeholder_name: str, value: str):
    if isinstance(it, dict):
        ordered_dict = OrderedDict()
        for k, v in it.items():
            ordered_dict[k] = _replace_placeholder(v, placeholder_name, value)
        return ordered_dict
    elif isinstance(it, list):
        return [_replace_placeholder(item, placeholder_name, value) for item in it]
    elif isinstance(it, str):
        placeholder = f"$${placeholder_name}$$"
        if placeholder == it:
            return value  # value placeholder
        else:
            return it.replace(placeholder, str(value))  # template
    else:
        return it


def expand_from_template(template: OrderedDict, placeholder_name: str, values: list) -> List[OrderedDict]:
    options = []
    for value in values:
        modified = _replace_placeholder(template, placeholder_name, value)
        options.append(modified)
    return options


def generate_options_from_template(template: OrderedDict) -> List[OrderedDict]:

    grid_search_settings: OrderedDict = template.pop("grid_search_settings") # template only

    # generate options like matrix
    options = [template]
    for grid_search_dim in grid_search_settings:
        current = options
        produced = []
        for key, values in grid_search_dim.items():
            for item in current:
                opt = expand_from_template(item, key, values)
                produced.extend(opt)
        options = produced

    return options


def train_from_template(root_path: str):
    arguments = parse_arguments()
    template_path = arguments.opt

    with open(template_path, mode='r') as f:
        template = yaml.load(f, Loader=ordered_yaml()[0])

    options = generate_options_from_template(template)
    print(f"Created {len(options)} options for grid search!")

    for i, opt in enumerate(options):
        print("================================")
        print(f"Training option {i + 1} {opt['name']}")
        print("================================")
        actual_opt = setup_env_and_update_options(
            opt=opt,
            launcher=arguments.launcher,
            auto_resume=arguments.auto_resume,
            debug=arguments.debug,
            force_yml=arguments.force_yml,
            root_path=root_path,
            is_train=True,
        )
        try:
            train_pipeline(actual_opt, opt_path=template_path, dump_real_option=True)
            print("================================")
            print(f"Training option {i + 1} Completed!")
            print("================================")
        except Exception:
            print("================================")
            print(f"Failed to train option {i + 1}!")
            import traceback
            traceback.print_exc()
            print("================================")
            sleep(8)


if __name__ == '__main__':
    train_from_template(ROOT_PATH)
