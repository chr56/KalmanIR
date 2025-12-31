import os

from torch import float32 as dtype_float32
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

import cv2
import numpy as np

SUPPORTED_TASKS = ['SR', 'denoising_color']

DEBUG = False


@DATASET_REGISTRY.register()
class EnhancedPairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration with enhanced augmentations.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(EnhancedPairedImageDataset, self).__init__()
        self.opt = opt

        self.task = opt['task'] if 'task' in opt else None
        self.noise = opt['noise'] if 'noise' in opt else 0

        assert self.task in SUPPORTED_TASKS
        assert self.task != 'denoising_color' or self.noise > 0

        self.scale = self.opt['scale']
        self.gt_size = self.opt['gt_size']
        self.lq_size = self.gt_size // self.scale

        self.convert_color_space = 'color' in self.opt and self.opt['color'] == 'y'

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl, self.task)

        ###################

        self.aug_opt = opt.get('augmentation', {})
        from .albumentations_transform import get_albumentations_transforms
        all_transforms = get_albumentations_transforms(
            aug_opt=self.aug_opt,
            task=self.task,
            gt_size=self.gt_size,
            lq_size=self.lq_size,
            noise=self.noise,
        )
        self.transforms_gt_train = all_transforms[0]
        self.transforms_lq_train = all_transforms[1]
        self.transform_lq_val = all_transforms[2]

    def _read_img_gt(self, index, convert_rgb: bool = True):
        gt_path = self.paths[index]['gt_path']
        img_bytes_gt = self.file_client.get(gt_path, 'gt')
        img_gt: np.ndarray = imfrombytes(img_bytes_gt, float32=False)  # Dimension order: HWC; channel order: BGR
        if convert_rgb:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        return gt_path, img_gt

    def _read_img_lq(self, index, has_lq: bool, convert_rgb: bool = True):
        lq_path = self.paths[index]['lq_path']
        if has_lq:
            img_bytes_lq = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes_lq, float32=False)  # Dimension order: HWC; channel order: BGR
            if convert_rgb:
                img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
        else:
            img_lq = None
        return lq_path, img_lq

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        lq_path, img_lq = self._read_img_lq(index, has_lq=False)

        gt_path, img_gt = self._read_img_gt(index)

        if self.opt['phase'] == 'train':
            img_gt = self.transforms_gt_train(image=img_gt)['image']
            img_lq = self.transforms_lq_train(image=img_gt)['image']
        else:  # val
            img_gt = img_gt
            img_lq = self.transform_lq_val.apply(img=img_gt)

        # color space transform
        if self.convert_color_space:
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        if DEBUG:
            from basicsr import imwrite
            imwrite(img_gt, f"images/img_{index}_gt.png")
            imwrite(img_lq, f"images/img_{index}_lq.png")

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        img_gt = img_gt.to(dtype_float32) / 255.
        img_lq = img_lq.to(dtype_float32) / 255.
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
