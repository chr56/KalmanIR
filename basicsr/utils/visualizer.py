import os
import os.path as osp
from typing import List, Literal

import torch

Transformation = Literal['b2d', 'd2b', 'flatten']


class Visualizer:
    ########################

    _static_instance: 'Visualizer' = None

    @classmethod
    def setup(cls, instance: 'Visualizer') -> None:
        cls._static_instance = instance

    @classmethod
    def instance(cls) -> 'Visualizer':
        if cls._static_instance is None:
            raise RuntimeError('Visualizer has not been initialized')
        return cls._static_instance

    ########################
    def __init__(self,
                 runs: str = None,
                 root_directory: str = 'results/image_visualizations',
                 implementation: str = 'torchvision',
                 mkdir: bool = True,
                 enabled: bool = True,
                 ):

        self.runs = runs or _default_name('%Y%m%d-%H%M%S')
        self.root_directory = root_directory

        self.directory = f"{self.root_directory}/{self.runs}"
        if mkdir:
            os.makedirs(osp.abspath(self.directory), exist_ok=True)

        if implementation == 'torchvision':
            self.visualize_and_save = _save_with_torchvision
        elif implementation == 'basic_sr':
            self.visualize_and_save = _save_with_basic_sr
        else:
            raise NotImplementedError(f"Unknown {implementation}")

        self.counter = Counter(0)
        self.prefix = None

        self.enabled = enabled

    def update_prefix(self, prefix):
        self.prefix = prefix

    def update_counter_as_prefix(self):
        self.prefix = self.counter()
        self.counter.bump()

    def clear_prefix(self):
        self.prefix = None

    def visualize(self,
                  tensor: torch.Tensor, name: str,
                  transforms: List[Transformation] = None,
                  ):
        if not self.enabled: return

        file_prefix = f"{self.prefix}_" if self.prefix is not None else ""
        path = f"{self.directory}/{file_prefix}{name}.png"
        tensor = tensor.detach().cpu()
        try:
            if transforms: tensor = transform_tensor(tensor, transforms)
            self.visualize_and_save(tensor, path)
        except IOError as e:
            print(f"Could not save to {path}: {e}")
        pass


class Counter:
    def __init__(self, value: int = 0):
        self.value = value

    def bump(self):
        self.value += 1

    def reset(self):
        self.value = 0

    def __call__(self):
        return f"{self.value:03d}"


def transform_tensor(tensor: torch.Tensor, transforms: List[Transformation]):
    for name in transforms:
        tensor = _perform_transform(tensor, name)
    return tensor


def _perform_transform(tensor, name: Transformation):
    if name == 'b2d':
        from .binary_transform import binary_to_decimal
        return binary_to_decimal(tensor)
    elif name == 'd2b':
        from .binary_transform import decimal_to_binary
        return decimal_to_binary(tensor * 255.)
    elif name == 'flatten':
        return _flatten_channels(tensor)
    else:
        return tensor


def _flatten_channels(tensor):
    from einops import rearrange
    tensor = rearrange(tensor, 'b c h w -> (b c) 1 h w')
    tensor = torch.sigmoid(tensor)
    return tensor


def _save_with_basic_sr(tensor, path, **kwargs):
    from basicsr.utils.img_util import tensor2img, imwrite
    img = tensor2img(tensor)
    imwrite(img, path, auto_mkdir=False)


def _save_with_torchvision(tensor, path, **kwargs):
    from torchvision.utils import save_image
    save_image(tensor, path, nrow=4)


def _default_name(time_format):
    from datetime import datetime
    return datetime.now().strftime(time_format)
