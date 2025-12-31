import cv2
from typing import Optional, List, Tuple

try:
    import os

    os.environ['NO_ALBUMENTATIONS_UPDATE'] = "1"  # disable update checking on module importing
    # noinspection PyPep8Naming
    import albumentations as A
except ImportError:
    import warnings

    warnings.warn(f"Albumentations is not installed!")
    pass


def get_albumentations_transforms(
        aug_opt: dict,
        task: str,
        gt_size: int,
        lq_size: int,
        noise: float,
) -> tuple:
    channel_shuffle: float = aug_opt.get('channel_shuffle', -1.)
    lq_color_jitter: float = aug_opt.get('lq_color_jitter', -1.)
    lq_blur: float = aug_opt.get('lq_blur', -1.)
    mean: Tuple[float] = aug_opt['mean'] if 'mean' in aug_opt else None
    std: Tuple[float] = aug_opt['std'] if 'std' in aug_opt else None

    _transforms_gt_train = [
        A.Rotate(p=0.7, limit=(-45, 45)),
        A.RandomCrop(height=gt_size, width=gt_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    if channel_shuffle > 0:
        _transforms_gt_train.append(A.ChannelShuffle(p=channel_shuffle))
    if mean is not None or std is not None:
        _transforms_gt_train.append(A.Normalize(mean=mean, std=std))

    transforms_gt_train = A.Compose(_transforms_gt_train)

    if task == 'denoising_color':
        _basic_degrade = A.GaussNoise(noise_scale_factor=noise / 255., p=1.0)
        _transforms_lq_train = []
        if lq_color_jitter > 0:
            _transforms_lq_train.append(
                A.ColorJitter(p=lq_color_jitter, brightness=0.1, contrast=0.2, saturation=0.1, hue=0.1)
            )
        if lq_blur > 0:
            _transforms_lq_train.append(
                A.GaussianBlur(p=lq_blur, blur_limit=int(lq_size / 8))
            )
        _transforms_lq_train.append(_basic_degrade)  # at last
        transforms_lq_train = A.Compose(transforms=_transforms_lq_train)
        transform_lq_val = _basic_degrade
    elif task == 'SR':
        _basic_degrade = A.Resize(height=lq_size, width=lq_size, interpolation=cv2.INTER_CUBIC)
        _transforms_lq_train = [_basic_degrade]  # at first
        if lq_color_jitter > 0:
            _transforms_lq_train.append(
                A.ColorJitter(p=lq_color_jitter, brightness=0.1, contrast=0.1, saturation=0.05)
            )
        transforms_lq_train = A.Compose(transforms=_transforms_lq_train)
        transform_lq_val = _basic_degrade
    else:
        raise NotImplementedError(f"Task {task} not implemented.")

    return transforms_gt_train, transforms_lq_train, transform_lq_val
