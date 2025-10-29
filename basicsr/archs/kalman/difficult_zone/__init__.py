from torch import nn


def build_difficult_zone_estimator(variant, dim: int, **kwargs) -> nn.Module:
    merge_ratio = kwargs.get('variant_difficult_zone_merge_ratio', 0.5)
    if variant == "original_v4":
        from .estimator_v4 import DifficultZoneEstimatorV4
        return DifficultZoneEstimatorV4(channel=dim)
    elif variant == "original_v6":
        from .estimator_v6 import DifficultZoneEstimatorV6
        return DifficultZoneEstimatorV6(channel=dim)
    elif variant == "deep_convolutional_v1":
        from .convolutional import DeepConvolutionalV1
        return DeepConvolutionalV1(channel=dim, merge_ratio=merge_ratio)
    elif variant == "deep_convolutional_v2":
        from .convolutional import DeepConvolutionalV2
        return DeepConvolutionalV2(channel=dim, merge_ratio=merge_ratio)
    elif variant == "deep_convolutional_v3":
        from .convolutional import DeepConvolutionalV3
        return DeepConvolutionalV3(channel=dim, merge_ratio=merge_ratio)
    elif variant == "multi_convolutional_v1":
        from .convolutional import MultiConvolutionalV1
        return MultiConvolutionalV1(channel=dim, hidden_channels=dim // 3)
    else:
        if variant:
            import warnings
            warnings.warn(f"Unknown difficult zone estimator variant `{variant}`, use default instead!")
        from .estimator_v6 import DifficultZoneEstimatorV6
        return DifficultZoneEstimatorV6(channel=dim)
