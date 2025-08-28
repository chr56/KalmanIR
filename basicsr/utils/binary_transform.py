# -*- coding: utf-8 -*-
# @Time    : 2024/8/17 15:27
# @Author  : stack Zhang
# @FileName: binary_transform.py
# @Email   : yanzhang1991@cqupt.edu.cn
import torch


def decimal_to_binary(input_tensor):
    #十进制(b, 3, h, w)  ->  二进制[b, 8, h, w]*3
    flag3 = False
    if len(input_tensor.shape) == 3:
        flag3 = True
        c, h, w = input_tensor.shape
        input_tensor = input_tensor.reshape(1, c, h, w)
    b, c, h, w = input_tensor.shape
    assert c == 3, "输入张量必须有 3 个通道"

    # 创建一个列表用于存储 3 个形状为 (b, 8, h, w) 的张量
    binary_tensors = []

    for channel in range(c):
        binary_tensor = torch.zeros((b, 8, h, w), dtype=torch.float32, device=input_tensor.device)
        for i in range(8):
            binary_tensor[:, i, :, :] = (input_tensor[:, channel, :, :].to(torch.uint8) >> i) & 1
        binary_tensors.append(binary_tensor)
    out = torch.cat(binary_tensors, dim=1)
    if flag3 is True:
        out = out[0, ...]
        flag3 = False
    return out

def binary_to_decimal(binary_tensors):
    # 二进制[b, 8, h, w]*3  ->  十进制(b, 3, h, w)
    flag3 = False
    if len(binary_tensors.shape) == 3:
        flag3 = True
        binary_tensors = binary_tensors.permute(2, 0, 1)
        c, h, w = binary_tensors.shape
        binary_tensors = binary_tensors.reshape(1, c, h, w)

    square = torch.tensor(
        [1., 2., 4., 8., 16., 32., 64., 128.], device=binary_tensors.device, dtype=binary_tensors.dtype
    ).reshape(1, 8, 1, 1)
    binary_tensors = torch.split(binary_tensors, 8, 1)
    decimal_tensor = []
    for i in range(3):
        decimal_tensor.append(torch.sum(square * binary_tensors[i], dim=1))
    decimal = torch.stack(decimal_tensor, dim=1)
    if flag3 is True:
        decimal = decimal[0, ...].permute(1, 2, 0)
        flag3 = False
    return decimal / 255.0

# def binary_to_decimal_mixup(binary_tensors, square):
#     # 二进制[b, 8, h, w]*3  ->  十进制(b, 3, h, w)
#     flag3 = False
#     if len(binary_tensors.shape) == 3:
#         flag3 = True
#         # binary_tensors = binary_tensors.permute(2, 0, 1)
#         c, h, w = binary_tensors.shape
#         binary_tensors = binary_tensors.reshape(1, c, h, w)
#
#     binary_tensors = torch.split(binary_tensors, 8, 1)
#     decimal_tensor = []
#     for i in range(3):
#         decimal_tensor.append(torch.sum(square * binary_tensors[i], dim=1))
#     decimal = torch.stack(decimal_tensor, dim=1)
#     if flag3 is True:
#         decimal = decimal[0, ...]#.permute(1, 2, 0)
#         flag3 = False
#     return decimal / 255.0