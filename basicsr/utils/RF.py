import torch

# Recursive Filter 实现
def rf(img, sigma_s, sigma_r, num_iterations=3, joint_image=None):
    I = img

    if joint_image is not None:
        J = joint_image
        if I.shape[2] != J.shape[2] or I.shape[3] != J.shape[3]:
            raise ValueError('Input and joint image must have equal width and height.')
    else:
        J = I

    b, num_joint_channels, h, w = J.shape

    # 计算 domain transform (参考论文方程 11)
    dIcdx = (J[:, :, :, 1:] - J[:, :, :, :-1]).to(I.device)  # 差分沿着宽度（列）
    dIcdy = (J[:, :, 1:, :] - J[:, :, :-1, :]).to(I.device)  # 差分沿着高度（行）

    dIdx = torch.zeros(b, h, w).to(I.device)
    dIdy = torch.zeros(b, h, w).to(I.device)

    # 计算 l1-norm 邻居像素间的距离
    for c in range(num_joint_channels):
        dIdx[:, :, 1:] += torch.abs(dIcdx[:, c, :, :])
        dIdy[:, 1:, :] += torch.abs(dIcdy[:, c, :, :])

    # 计算 domain transform 的水平和垂直导数
    dHdx = 1 + sigma_s / sigma_r * dIdx
    dVdy = 1 + sigma_s / sigma_r * dIdy
    dVdy = dVdy.permute(0, 2, 1)
    # 执行滤波
    N = num_iterations
    F = I
    sigma_H = sigma_s

    for i in range(num_iterations):
        # 计算每次迭代的 sigma 值（参考论文方程 14）
        sigma_H_i = sigma_H * torch.sqrt(torch.tensor(3.0, device=I.device)) * 2 ** (N - (i + 1)) / torch.sqrt(torch.tensor(4.0, device=I.device) ** N - 1)

        # 水平滤波
        F = TransformedDomainRecursiveFilter_Horizontal(F, dHdx, sigma_H_i)
        F = F.permute(0, 1, 3, 2)

        # 垂直滤波
        F = TransformedDomainRecursiveFilter_Horizontal(F, dVdy, sigma_H_i)
        F = F.permute(0, 1, 3, 2)

    return F

# 水平 Recursive Filter 实现
def TransformedDomainRecursiveFilter_Horizontal(I, D, sigma):
    # Feedback 系数 (参考论文附录)
    a = torch.exp(-torch.sqrt(torch.sqrt(torch.tensor(2.0, device=I.device)) / sigma))
    F = I

    V = torch.pow(a, D).to(I.device)

    b, num_channels, h, w = I.shape

    # Left -> Right 滤波
    for c in range(num_channels):
        F_tmp = torch.zeros(b, num_channels, w, h).to(I.device)
        F_tmp[:, c] = F[:, c].permute(0, 2, 1)
        for i in range(1, w):
            F_tmp[:, c, i] += V[:, :, i] * (F_tmp[:, c, i - 1] - F_tmp[:, c, i])
        F[:, c] = F_tmp[:, c].permute(0, 2, 1)

    # Right -> Left 滤波
    for c in range(num_channels):
        F_tmp = torch.zeros(b, num_channels, w, h).to(I.device)
        F_tmp[:, c] = F[:, c].permute(0, 2, 1)
        for i in range(w - 2, -1, -1):
            F_tmp[:, c, i] += V[:, :, i + 1] * (F_tmp[:, c, i + 1] - F_tmp[:, c, i])
        F[:, c] = F_tmp[:, c].permute(0, 2, 1)

    return F