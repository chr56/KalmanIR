from torch import nn


def build_gain_calculator_for_v4(mode, dim) -> nn.Module:
    if mode == "ss2d":
        from .mamba import KalmanGainCalculatorMambaSimple
        return KalmanGainCalculatorMambaSimple(dim)
    elif mode == "block":
        from .mamba import KalmanGainCalculatorMambaBlock
        return KalmanGainCalculatorMambaBlock(dim)
    else:
        from .original import KalmanGainCalculatorV0
        return KalmanGainCalculatorV0(dim)


def build_gain_calculator(variant, dim: int, seq_length: int, **kwargs) -> nn.Module:
    uncertainty_update_ratio = kwargs.get('variant_gain_calculation_uncertainty_update_ratio', 0.5)
    if variant == "linear_convolutional_multiple_channels":
        from .original import LinearConvolutionalMultipleChannels
        return LinearConvolutionalMultipleChannels(channel=dim, seq_length=seq_length)
    elif variant == "simple_convolutional_multiple_channels":
        from .original import SimpleConvolutionalMultipleChannels
        return SimpleConvolutionalMultipleChannels(channel=dim, seq_length=seq_length)
    elif variant == "deep_convolutional_multiple_channels_v1":
        from .convolutional import DeepConvolutionalMultipleChannelsV1
        return DeepConvolutionalMultipleChannelsV1(channel=dim, seq_length=seq_length,
                                                   uncertainty_update_ratio=uncertainty_update_ratio)
    elif variant == "deep_convolutional_multiple_channels_v2":
        from .convolutional import DeepConvolutionalMultipleChannelsV2
        return DeepConvolutionalMultipleChannelsV2(channel=dim, seq_length=seq_length,
                                                   uncertainty_update_ratio=uncertainty_update_ratio)
    elif variant == "deep_convolutional_multiple_channels_v3a":
        from .convolutional import DeepConvolutionalMultipleChannelsV3a
        return DeepConvolutionalMultipleChannelsV3a(channel=dim, seq_length=seq_length,
                                                    uncertainty_update_ratio=uncertainty_update_ratio)
    elif variant == "deep_convolutional_multiple_channels_v3b":
        from .convolutional import DeepConvolutionalMultipleChannelsV3b
        return DeepConvolutionalMultipleChannelsV3b(channel=dim, seq_length=seq_length,
                                                    uncertainty_update_ratio=uncertainty_update_ratio)
    elif variant == "deep_convolutional_multiple_channels_v3c":
        from .convolutional import DeepConvolutionalMultipleChannelsV3c
        return DeepConvolutionalMultipleChannelsV3c(channel=dim, seq_length=seq_length,
                                                    uncertainty_update_ratio=uncertainty_update_ratio)
    elif variant == "deep_convolutional_multiple_channels_v4":
        from .convolutional import DeepConvolutionalMultipleChannelsV4
        return DeepConvolutionalMultipleChannelsV4(channel=dim, seq_length=seq_length,
                                                   uncertainty_update_ratio=uncertainty_update_ratio)
    elif variant == "deep_convolutional_multiple_channels_v5":
        from .convolutional import DeepConvolutionalMultipleChannelsV5
        return DeepConvolutionalMultipleChannelsV5(channel=dim, seq_length=seq_length,
                                                   uncertainty_update_ratio=uncertainty_update_ratio)
    elif variant == "deep_convolutional_multiple_channels_v5e":
        from .convolutional import DeepConvolutionalMultipleChannelsV5e
        return DeepConvolutionalMultipleChannelsV5e(channel=dim, seq_length=seq_length,
                                                    uncertainty_update_ratio=uncertainty_update_ratio)
    elif variant == "complex_convolutional_multiple_channels":
        from .original import ComplexConvolutionalMultipleChannels
        return ComplexConvolutionalMultipleChannels(channel=dim, seq_length=seq_length)
    else:
        if variant:
            import warnings
            warnings.warn(f"Unknown gain calculator variant `{variant}`, use default instead!")
        from .original import SimpleConvolutionalMultipleChannels
        return SimpleConvolutionalMultipleChannels(channel=dim, seq_length=seq_length)
