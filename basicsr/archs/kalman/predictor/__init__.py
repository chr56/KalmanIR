import torch
from torch import nn as nn


def build_predictor(mode, dim, seq_length) -> nn.Module:
    if mode == "convolutional":
        from .original import KalmanPredictorV0
        return KalmanPredictorV0(dim=dim)
    elif mode == "deep_convolutional_v1":
        from .convolutional import KalmanPredictorDeepConvolutionalV1
        return KalmanPredictorDeepConvolutionalV1(dim=dim)
    elif mode == "deep_convolutional_v2":
        from .convolutional import KalmanPredictorDeepConvolutionalV2
        return KalmanPredictorDeepConvolutionalV2(dim=dim)
    elif mode == "mamba_adjustment":
        from .mamba import KalmanPredictorMambaAdjustment
        return KalmanPredictorMambaAdjustment(dim=dim)
    elif mode == "mamba_latent_adjustment":
        from .mamba import KalmanPredictorMambaLatentAdjustment
        return KalmanPredictorMambaLatentAdjustment(dim=dim)
    else:
        if mode:
            import warnings
            warnings.warn(f"Unknown kalman preditor mode `{mode}`, using default!")
        from .original import KalmanPredictorV0
        return KalmanPredictorV0(dim=dim)