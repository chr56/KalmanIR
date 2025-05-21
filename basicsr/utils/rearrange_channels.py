import torch


def rearrange_channels(tensor):
    B, C, H, W = tensor.shape
    assert C % 3 == 0, "输入通道数量必须为3的整数倍"
    C = int(C)
    index = torch.zeros(C, dtype=torch.int64)
    c = int(C / 3)
    for i in range(c):
        index[3 * i] = i
        index[3 *i + 1] = i + c
        index[3 * i + 2] = i + 2 * c
    index = index.view(1, C, 1, 1).expand(B, C, H, W)

    rearranged_tensor = torch.gather(tensor, dim=1, index=index)

    return rearranged_tensor



# tensor = torch.randn(2, 27, 4, 4)
#
# rearranged_tensor = rearrange_channels(tensor)
# print(rearranged_tensor.shape)
#
# B, H, W = 1, 2, 2
#
# tensor = torch.arange(27 * H * W).view(B, 27, H, W).float()
# print("Original Tensor:\n", tensor)
#
# rearranged_tensor = rearrange_channels(tensor.clone())
# print("Rearranged Tensor:\n", rearranged_tensor)







