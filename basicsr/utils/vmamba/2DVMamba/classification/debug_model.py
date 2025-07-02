import torch
def permute(x, hw_shape, B, direction_Bs, permute_B=False):
    # x: B, L, D(E)
    # B: B, d_state, L
    # direction_Bs: 4, dstate
    H, W = hw_shape
    BB, L, D = x.shape
    x = x.reshape(BB, H, W, D)

    x_1 = x.permute(0, 3, 1, 2)  # [B, L, H, W]
    HW_1 = (H, W)

    x_2 = x_1.permute(0, 1, 3, 2)  # [B, L, W, H]
    HW_2 = (W, H)

    x_1 = x_1.flatten(2)
    x_2 = x_2.flatten(2)

    x_3 = x_1.flip(-1)
    HW_3 = HW_1
    x_4 = x_2.flip(-1)
    HW_4 = HW_2

    if permute_B:
        B = B.reshape(B.shape[0], B.shape[1], H, W)
        B1 = B.flatten(2)
        B2 = B.permute(0, 1, 3, 2).flatten(2)
        B3 = B1.flip(-1)
        B4 = B2.flip(-1)
        Bs = [B1, B2, B3, B4]
    else:
        Bs = [B, B, B, B]

    dBs = [db[None, :, None] for db in direction_Bs]

    return [x_1, x_2, x_3, x_4], [HW_1, HW_2, HW_3, HW_4], Bs, dBs

def unpermute_and_sum(ys, H, W):
    # ys list of 4 [B, D, L]
    ys0 = ys[0]
    ys1 = ys[1]
    ys2 = ys[2].flip(-1)
    ys3 = ys[3].flip(-1)

    ys02 = ys0 + ys2
    ys13 = ys1 + ys3
    ys13 = ys13.reshape(ys13.shape[0], ys13.shape[1], W, H)
    ys13 = ys13.permute(0, 1, 3, 2).flatten(2)
    ys_out = ys02 + ys13
    ys_out = ys_out.permute(0, 2, 1)
    return ys_out


if __name__ == '__main__':
    x = torch.range(0, 5).reshape(1, 6, 1)
    hw_shape = (2, 3)
    B = torch.range(0, 5).reshape(1, 1, 6)
    direction_Bs = torch.range(0, 3).reshape(4, 1)

    xs, HWs, Bs, dBs = permute(x, hw_shape, B, direction_Bs, permute_B=True)

    ys_out = unpermute_and_sum(xs, *hw_shape)

    print(xs)
    print(HWs)
    print(Bs)
    print(dBs)
    print(ys_out)