import cv2
import math
import numpy as np
import os
import datetime
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')


def calculate_and_padding_image(img, patch_size=200):
    _, _, raw_h, raw_w = img.size()

    num_patch_h = raw_h // patch_size + 1  # number of horizontal cut sections
    num_patch_w = raw_w // patch_size + 1  # number of vertical cut sections

    # padding
    padding_h, padding_w = 0, 0
    if raw_h % num_patch_h != 0:
        padding_h = num_patch_h - raw_h % num_patch_h
    if raw_w % num_patch_w != 0:
        padding_w = num_patch_w - raw_w % num_patch_w
    img = torch.nn.functional.pad(img, (0, padding_w, 0, padding_h), 'reflect')

    _, _, H, W = img.size()
    patch_h = H // num_patch_h  # height of each patch
    patch_w = W // num_patch_w  # width of each patch
    col = H // patch_h  # number of patches in height
    row = W // patch_w  # number of patches in width
    # overlapping
    shave_h = patch_h // 10
    shave_w = patch_w // 10
    return img, col, row, padding_h, padding_w, patch_h, patch_w, shave_h, shave_w


def calculate_borders_for_chopping(
        total_col: int, total_row: int,
        patch_h: int, patch_w: int,
        shave_h: int, shave_w: int,
) -> list:
    slices = []  # list of partition borders
    for i in range(total_col):
        for j in range(total_row):
            if i == 0 and i == total_col - 1:
                h_range = slice(i * patch_h, (i + 1) * patch_h)
            elif i == 0:
                h_range = slice(i * patch_h, (i + 1) * patch_h + shave_h)
            elif i == total_col - 1:
                h_range = slice(i * patch_h - shave_h, (i + 1) * patch_h)
            else:
                h_range = slice(i * patch_h - shave_h, (i + 1) * patch_h + shave_h)
            if j == 0 and j == total_row - 1:
                w_range = slice(j * patch_w, (j + 1) * patch_w)
            elif j == 0:
                w_range = slice(j * patch_w, (j + 1) * patch_w + shave_w)
            elif j == total_row - 1:
                w_range = slice(j * patch_w - shave_w, (j + 1) * patch_w)
            else:
                w_range = slice(j * patch_w - shave_w, (j + 1) * patch_w + shave_w)
            box = (h_range, w_range)
            slices.append(box)
    return slices


def recover_from_patches(
        patches,
        total_col: int, total_row: int,
        batch_size: int, channel: int, lq_width: int, lq_height: int, sr_scale,
        patch_h: int, patch_w: int, shave_h: int, shave_w: int,
) -> torch.Tensor:
    _img = torch.zeros(batch_size, channel, lq_height * sr_scale, lq_width * sr_scale)
    for i in range(total_col):
        for j in range(total_row):
            target_h_range = slice(i * patch_h * sr_scale, (i + 1) * patch_h * sr_scale)
            target_w_range = slice(j * patch_w * sr_scale, (j + 1) * patch_w * sr_scale)
            if i == 0:
                h_range = slice(0, patch_h * sr_scale)
            else:
                h_range = slice(shave_h * sr_scale, (shave_h + patch_h) * sr_scale)
            if j == 0:
                w_range = slice(0, patch_w * sr_scale)
            else:
                w_range = slice(shave_w * sr_scale, (shave_w + patch_w) * sr_scale)
            _img[..., target_h_range, target_w_range] = patches[i * total_row + j][..., h_range, w_range]
    return _img


def recover_from_patches_and_remove_paddings(
        patches,
        total_col: int, total_row: int,
        batch_size: int, channel: int, lq_width: int, lq_height: int, sr_scale,
        patch_h: int, patch_w: int, shave_h: int, shave_w: int,
        padding_h: int, padding_w: int, scale,
) -> torch.Tensor:
    recovered = recover_from_patches(
        patches,
        total_col, total_row,
        batch_size, channel, lq_width, lq_height, sr_scale,
        patch_h, patch_w, shave_h, shave_w,
    )
    _, _, h, w = recovered.size()
    recovered = recovered[:, :, 0:h - padding_h * scale, 0:w - padding_w * scale]
    return recovered


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]

def dump_images(sr, hr, save_directory: str):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    pil_transform = ToPILImage()
    p0 = hr[0]
    p1 = sr[0]
    plc0 = pil_transform(p0)
    plc1 = pil_transform(p1)
    save_path = os.path.join(save_directory, f'{formatted_time}_HR.png')
    save_path1 = os.path.join(save_directory, f'{formatted_time}_SR.png')
    plc0.save(save_path)
    plc1.save(save_path1)
