import os
import torch


class Visualizer:
    def __init__(self,
                 implementation: str = 'torchvision',
                 root_directory: str = 'results/image_visualizations',
                 time_format: str = '%Y%m%d-%H%M%S',
                 mkdir: bool = True,
                 ):
        from datetime import datetime
        self.runs = datetime.now().strftime(time_format)
        self.root_directory = root_directory

        self.directory = f"{self.root_directory}/{self.runs}"
        if mkdir:
            os.makedirs(os.path.abspath(self.directory), exist_ok=True)

        if implementation == 'torchvision':
            self.visualize_and_save = self.save_with_torchvision
        elif implementation == 'basic_sr':
            self.visualize_and_save = self.save_with_basic_sr
        else:
            raise NotImplementedError(f"Unknown {implementation}")

        self.counter = 0

    def visualize_images(self,
                         tensor: torch.Tensor, name: str,
                         without_counter: bool = False,
                         bump_counter: bool = False,
                         transform: str = None,
                         ):
        if without_counter:
            path = f"{self.directory}/{name}.png"
        else:
            path = f"{self.directory}/{self.counter:03d}_{name}.png"
        tensor = tensor.detach().cpu()
        try:
            if transform:
                tensor = self._transform(tensor, transform)
            self.visualize_and_save(tensor, path)
        except IOError:
            print(f"Could not save {path}")
            pass
        pass

        if bump_counter:
            self.counter += 1

    def visualize_tensor(self,
                         tensor: torch.Tensor, name: str,
                         without_counter: bool = False,
                         bump_counter: bool = False,
                         transform: str = None,
                         ):
        if without_counter:
            path = f"{self.directory}/{name}.png"
        else:
            path = f"{self.directory}/{self.counter:03d}_{name}.png"
        tensor = tensor.detach().cpu()
        try:
            if transform:
                tensor = self._transform(tensor, transform)
            tensor = self.flatten_channels(tensor)
            self.visualize_and_save(tensor, path)
        except IOError:
            print(f"Could not save {path}")
            pass
        pass

        if bump_counter:
            self.counter += 1

    @staticmethod
    def _transform(tensor, name: str):
        if name == 'b2d':
            from .binary_transform import binary_to_decimal
            return binary_to_decimal(tensor)
        elif name == 'd2b':
            from .binary_transform import decimal_to_binary
            return decimal_to_binary(tensor * 255.)
        else:
            return tensor

    @staticmethod
    def save_with_basic_sr(tensor, path, **kwargs):
        from basicsr.utils.img_util import tensor2img, imwrite
        img = tensor2img(tensor)
        imwrite(img, path, auto_mkdir=False)

    @staticmethod
    def save_with_torchvision(tensor, path, **kwargs):
        from torchvision.utils import save_image
        save_image(tensor, path, nrow=4)

    @staticmethod
    def flatten_channels(tensor):
        from einops import rearrange
        tensor = rearrange(tensor, 'b c h w -> (b c) 1 h w')
        tensor = torch.sigmoid(tensor)
        return tensor

    def bump_counter(self):
        self.counter += 1

    def reset_counter(self):
        self.counter = 0
