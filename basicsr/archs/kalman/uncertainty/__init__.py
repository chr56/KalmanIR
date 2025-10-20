from torch import nn

def build_uncertainty_estimator(variant, dim: int, seq_length: int, **kwargs) -> nn.Module:
    update_ratio = kwargs.get('variant_uncertainty_update_ratio', 0.5)
    if variant == "skipped":
        from .misc import DummyUncertaintyEstimator
        return DummyUncertaintyEstimator()
    elif variant == "mamba_recursive_state_adjustment_v1":
        from .recursive_mamba import MambaRecursiveStateAdjustmentV1
        return MambaRecursiveStateAdjustmentV1(seq_length, dim)
    elif variant == "mamba_recursive_state_adjustment_v2":
        from .recursive_mamba import MambaRecursiveStateAdjustmentV2
        return MambaRecursiveStateAdjustmentV2(seq_length, dim)
    elif variant == "mamba_recursive_state_adjustment_v2l":
        from .recursive_mamba import MambaRecursiveStateAdjustmentV2Lite
        return MambaRecursiveStateAdjustmentV2Lite(seq_length, dim)
    elif variant == "mamba_recursive_state_adjustment_v3":
        from .recursive_mamba import MambaRecursiveStateAdjustmentV3
        return MambaRecursiveStateAdjustmentV3(seq_length, dim)
    elif variant == "mamba_recursive_state_adjustment_v4":
        from .recursive_mamba import MambaRecursiveStateAdjustmentV4
        return MambaRecursiveStateAdjustmentV4(seq_length, dim)
    elif variant == "mamba_recursive_state_adjustment_v5":
        from .recursive_mamba import MambaRecursiveStateAdjustmentV5
        return MambaRecursiveStateAdjustmentV5(seq_length, dim)
    elif variant == "mamba_recursive_state_adjustment_v6":
        from .recursive_mamba import MambaRecursiveStateAdjustmentV6
        return MambaRecursiveStateAdjustmentV6(seq_length, dim)
    elif variant == "mamba_recursive_state_adjustment_v6x":
        from .recursive_mamba import MambaRecursiveStateAdjustmentV6X
        return MambaRecursiveStateAdjustmentV6X(seq_length, dim)
    elif variant == "recursive_convolutional_v1":
        from .recursive_convolutional import RecursiveConvolutionalV1
        return RecursiveConvolutionalV1(seq_length, dim)
    elif variant == "recursive_convolutional_v2":
        from .recursive_convolutional import RecursiveConvolutionalV2
        return RecursiveConvolutionalV2(seq_length, dim)
    elif variant == "recursive_convolutional_v3":
        from .recursive_convolutional import RecursiveConvolutionalV3
        return RecursiveConvolutionalV3(seq_length, dim)
    elif variant == "simple_recursive_convolutional_v1":
        from .recursive_convolutional import SimpleRecursiveConvolutionalV1
        return SimpleRecursiveConvolutionalV1(seq_length, dim, update_ratio)
    else:
        raise ValueError(f"Unsupported variant: {variant}")


def build_uncertainty_estimator_for_v4(mode, dim, seq_length) -> nn.Module:
    if mode == "iterative_convolutional":
        from .v4_convolutional import UncertaintyEstimatorIterativeConvolutional
        return UncertaintyEstimatorIterativeConvolutional(dim)
    elif mode == "iterative_narrow_mamba_block":
        from .v4_mamba import UncertaintyEstimatorIterativeNarrowMambaBlock
        return UncertaintyEstimatorIterativeNarrowMambaBlock(seq_length, dim)
    elif mode == "iterative_wide_mamba_block":
        from .v4_mamba import UncertaintyEstimatorIterativeWideMambaBlock
        return UncertaintyEstimatorIterativeWideMambaBlock(seq_length, dim)
    elif mode == "one_decoder_layer":
        from .v4_transformer import UncertaintyEstimatorOneDecoderLayer
        return UncertaintyEstimatorOneDecoderLayer(seq_length, dim, seq_length)
    elif mode == "iterative_decoder_layer":
        from .v4_transformer import UncertaintyEstimatorIterativeDecoderLayer
        return UncertaintyEstimatorIterativeDecoderLayer(dim, seq_length)
    elif mode == "one_cross_attention":
        from .v4_transformer import UncertaintyEstimatorOneCrossAttention
        return UncertaintyEstimatorOneCrossAttention(seq_length, dim, seq_length)
    elif mode == "iterative_cross_attention":
        from .v4_transformer import UncertaintyEstimatorIterativeCrossAttention
        return UncertaintyEstimatorIterativeCrossAttention(dim, 3)
    elif mode == "iterative_convolutional_cross_attention":
        from .v4_transformer import UncertaintyEstimatorIterativeConvolutionalCrossAttention
        return UncertaintyEstimatorIterativeConvolutionalCrossAttention(dim, 3)
    elif mode == "iterative_mamba_error_estimation":
        from .v4_mamba import UncertaintyEstimatorIterativeMambaErrorEstimation
        return UncertaintyEstimatorIterativeMambaErrorEstimation(seq_length, dim)
    elif mode == "iterative_mamba_error_estimation_v2":
        from .v4_mamba import UncertaintyEstimatorIterativeMambaErrorEstimationV2
        return UncertaintyEstimatorIterativeMambaErrorEstimationV2(seq_length, dim)
    else:
        if mode:
            import warnings
            warnings.warn(f"Unknown uncertainty estimator mode `{mode}`, using default!")
        from .v4_convolutional import UncertaintyEstimatorIterativeConvolutional
        return UncertaintyEstimatorIterativeConvolutional(dim)
