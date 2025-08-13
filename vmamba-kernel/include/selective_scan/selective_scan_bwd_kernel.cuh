/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once


#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <ATen/cuda/Atomic.cuh>  // For atomicAdd on complex

#include <math_constants.h>  // cudart header for CUDART_NAN_F

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

#include "scan/block_scan.cuh"
#include "selective_scan/global.cuh"
#include "selective_scan/selective_scan.cuh"
#include "selective_scan/selective_scan_common.cuh"
#include "selective_scan/static_switch.cuh"



template <typename scalar_t>
__device__ __forceinline__ scalar_t conj(scalar_t x);

template <>
__device__ __forceinline__ float conj<float>(float x) { return x; }

template <>
__device__ __forceinline__ complex_t conj<complex_t>(complex_t x) { return std::conj(x); }



template <int kBlockDim_, int kThreadSpan_, bool kIsEvenLen_,
        bool kIsVariableB_, bool kIsVariableC_,
        bool kDeltaSoftplus_, bool kHasZ_, typename input_t_, typename weight_t_, typename output_t_>
struct SelectiveScanBwdKernelTraits
{
    static constexpr int kThreadSpan = kThreadSpan_;
    // static_assert(kNItems_ % 4 == 0);

    using input_t = input_t_;
    using weight_t = weight_t_;
    using output_t = output_t_;

    static constexpr int kBlockDim = kBlockDim_;
    // static constexpr int kNThreads = kNThreads_;
    // static constexpr int kNItems = kNItems_;

    // static constexpr int kNBytes = sizeof(input_t);
    // static_assert(kNBytes == 2 || kNBytes == 4);
    // static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    // static_assert(kNItems % kNElts == 0);
    // static constexpr int kNLoads = kNItems / kNElts;

    static constexpr bool kIsComplex = std::is_same_v<weight_t, complex_t>;
    // static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kDeltaSoftplus = kDeltaSoftplus_;
    static constexpr bool kHasZ = kHasZ_;

    // Original mamba.
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads with float improves occupancy.
    // For complex this would lead to massive register spilling, so we keep it at 2.
    // static constexpr int kMinBlocks = (kBlockDim * kBlockDim) < 128 && !kIsComplex ? 3 : 2;

    // ndmamba:
    static constexpr int kMinBlocks = (kBlockDim * kBlockDim) < 128 ? 3 : 1;

    using scan_t = std::conditional_t<!kIsComplex, float2, float4>;

//    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
//    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
//    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, !kIsComplex ? kNItems : kNItems *
//                                                                                        2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
//    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, !kIsComplex ? kNLoads : kNLoads *
//                                                                                         2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
//    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
//    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_WARP_TRANSPOSE>;
//    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
//    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
//    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
//    using BlockReverseScanT = BlockReverseScan<scan_t, kNThreads>;

    // typename T, int BLOCK_DIM_X, BlockReduceAlgorithm ALGORITHM = BLOCK_REDUCE_WARP_REDUCTIONS, int BLOCK_DIM_Y = 1
    using BlockReduce = cub::BlockReduce<scan_t, kBlockDim, cub::BLOCK_REDUCE_WARP_REDUCTIONS, kBlockDim>;
    using BlockReduceFloat = cub::BlockReduce<float, kBlockDim, cub::BLOCK_REDUCE_WARP_REDUCTIONS, kBlockDim>;
//    using BlockReduceComplexT = cub::BlockReduce<complex_t, kNThreads>;

//    using BlockExchangeT = cub::BlockExchange<float, kNThreads, !kIsComplex ? kNItems : kNItems * 2>;

//    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
//                                                   sizeof(typename BlockLoadVecT::TempStorage),
//                                                   (int(kIsVariableB) + int(kIsVariableC)) *
//                                                   sizeof(typename BlockLoadWeightT::TempStorage),
//                                                   (int(kIsVariableB) + int(kIsVariableC)) *
//                                                   sizeof(typename BlockLoadWeightVecT::TempStorage),
//                                                   sizeof(typename BlockStoreT::TempStorage),
//                                                   sizeof(typename BlockStoreVecT::TempStorage)});
//    static constexpr int kSmemExchangeSize =
//            (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockExchangeT::TempStorage);
//    static constexpr int kSmemReduceSize = sizeof(typename BlockReduceT::TempStorage);
//    static constexpr int kSmemSize =
//            kSmemIOSize + kSmemExchangeSize + kSmemReduceSize + sizeof(typename BlockScanT::TempStorage) +
//            sizeof(typename BlockReverseScanT::TempStorage);

    static constexpr int kSegLen = 8;
    using BlockScan = ndmamba::SegBlockScan<scan_t, kSegLen, kBlockDim, kBlockDim>;

    static constexpr int kACacheSize = (kBlockDim * kThreadSpan + 1) * (kBlockDim * kThreadSpan + 1) * sizeof(float);

    static constexpr int kSmemSize =
            sizeof(typename BlockScan::TempStorage) +
            sizeof(typename BlockReduce::TempStorage) +
            kACacheSize;
};


template <typename Kernel>
__global__ __launch_bounds__(Kernel::kBlockDim * Kernel::kBlockDim, Kernel::kMinBlocks)
#if false
void selective_scan_bwd_kernel(SSMParamsBwd params) {}
#else
void selective_scan_bwd_kernel(SSMParamsBwd params)
{
    constexpr bool kIsComplex = Kernel::kIsComplex;
    constexpr bool kIsVariableB = Kernel::kIsVariableB;
    constexpr bool kIsVariableC = Kernel::kIsVariableC;
    constexpr bool kDeltaSoftplus = Kernel::kDeltaSoftplus;
    constexpr bool kHasZ = Kernel::kHasZ;

    constexpr int kBlockDim = Kernel::kBlockDim;
    constexpr int kThreadSpan = Kernel::kThreadSpan;

    using input_t = typename Kernel::input_t;
    using weight_t = typename Kernel::weight_t;
    using scan_t = typename Kernel::scan_t;
    using output_t = typename Kernel::output_t;

    // Shared memory Layout:
    // sizeof(typename Kernel::BlockScan::TempStorage)
    // sizeof(typename Kernel::BlockReduce::TempStorage)
    // Kernel::kACacheSize
    // sizeof(float) * params.dstate
    // sizeof(float) * params.dstate
    extern __shared__ char smem_[];

    auto & scanTempStorage = reinterpret_cast<typename Kernel::BlockScan::TempStorage &>(smem_);

    auto & reduceTempStorage = *reinterpret_cast<typename Kernel::BlockReduce::TempStorage *>(
            smem_ + sizeof(typename Kernel::BlockScan::TempStorage)
    );

    auto & reduceFloatTempStorage = *reinterpret_cast<typename Kernel::BlockReduceFloat::TempStorage *>(
            smem_ + sizeof(typename Kernel::BlockScan::TempStorage)
    );

    auto smem_delta_a_exp = reinterpret_cast<float *>(
            smem_ +
            sizeof(typename Kernel::BlockScan::TempStorage) +
            sizeof(typename Kernel::BlockReduce::TempStorage)
    );
    constexpr int kLdSmemDeltaAExp = kBlockDim * kThreadSpan + 1;

    // sizeof(float) * params.dstate
    auto * smem_da = reinterpret_cast<weight_t *>(reinterpret_cast<char *>(smem_delta_a_exp) + Kernel::kACacheSize);

    // sizeof(float) * params.dstate
    auto smem_dbc = reinterpret_cast<weight_t *>(smem_da) + params.dstate;

    // smem_delta_a_exp holds the delta_a_exp values for
    // (1) the current chunk (of size (kBlockDim * kThreadSpan, kBlockDim * kThreadSpan,)), and
    // (2) the first row of the previous chunk (used for rev horizontal scans), and
    // (3) the first col of the previous chunk (used for rev vertical scans).
    // smem_delta_a_exp itself is of size (kLdSmemDeltaAExp=kBlockDim * kThreadSpan + 1, kLdSmemDeltaAExp,).
    // The top-left submatrix of shape (kBlockDim * kThreadSpan, kBlockDim * kThreadSpan,) holds (1),
    // the rightmost column holds (2) and the bottommost row holds (3).
    auto deltaAExpCache = [smem_delta_a_exp, kLdSmemDeltaAExp](int yy, int xx) -> float &
    {
        const int cache_x = threadIdx.x * kThreadSpan + xx;
        const int cache_y = threadIdx.y * kThreadSpan + yy;
        return smem_delta_a_exp[cache_y * kLdSmemDeltaAExp + cache_x];
    };

    auto deltaAExpCacheShifted = [smem_delta_a_exp, kLdSmemDeltaAExp, kBlockDim]
            (int yy, int xx, ndmamba::ScanDir scanDir) -> float
    {
        int cache_x = threadIdx.x * kThreadSpan + xx;
        int cache_y = threadIdx.y * kThreadSpan + yy;
        constexpr int kCacheValidSize = kBlockDim * kThreadSpan;

        switch (scanDir)
        {
            case ndmamba::kHorizontalReversed:
            {
                ++cache_x;
                break;
            }
            case ndmamba::kVerticalReversed:
            {
                ++cache_y;
                break;
            }
            default:
            {
                printf("CUDA Kernel - NotImplementedError: "
                       "unsupported ndmamba::ScanDir in lambda deltaAExpCacheShifted");
                __trap();
            }
        }

        if (cache_x <= kCacheValidSize && cache_y <= kCacheValidSize)
        {
            return smem_delta_a_exp[cache_y * kLdSmemDeltaAExp + cache_x];
        }
        else
        {
            return CUDART_NAN_F;
        }
    };

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);

    input_t * u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
                  + dim_id * params.u_d_stride;
    input_t * delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
                      + dim_id * params.delta_d_stride;
    output_t * dout = reinterpret_cast<output_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride
                     + dim_id * params.dout_d_stride;
    weight_t * A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;
    weight_t * B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * params.B_d_stride;
    input_t * Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride +
                     group_id * params.B_group_stride;
    weight_t * C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * params.C_d_stride;
    input_t * Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride +
                     group_id * params.C_group_stride;
    weight_t * dA = reinterpret_cast<weight_t *>(params.dA_ptr) + dim_id * params.dA_d_stride;
    weight_t * dB = reinterpret_cast<weight_t *>(params.dB_ptr)
                    + (!kIsVariableB ? dim_id * params.dB_d_stride :
                       batch_id * (!kIsComplex ? params.dB_batch_stride : params.dB_batch_stride / 2) +
                       group_id * params.dB_group_stride);
    weight_t * dC = reinterpret_cast<weight_t *>(params.dC_ptr)
                    + (!kIsVariableC ? dim_id * params.dC_d_stride :
                       batch_id * (!kIsComplex ? params.dC_batch_stride : params.dC_batch_stride / 2) +
                       group_id * params.dC_group_stride);
    float * dD = params.dD_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dD_ptr) + dim_id;
    float D_val = params.D_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.D_ptr)[dim_id];
    float * ddelta_bias =
            params.ddelta_bias_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.ddelta_bias_ptr) + dim_id;
    float delta_bias = params.delta_bias_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id];

    input_t * du = reinterpret_cast<input_t *>(params.du_ptr)
                   + batch_id * params.du_batch_stride
                   + dim_id * params.du_d_stride;

    input_t * ddelta = reinterpret_cast<input_t *>(params.ddelta_ptr)
                       + batch_id * params.ddelta_batch_stride
                       + dim_id * params.ddelta_d_stride;

    // Size of a square chunk (when there are multiple chunks).
    // Invalid when there's only one chunk.
    constexpr int kChunkDim = ndmamba::kMaxDimPerBlock;

    const int numChunksAcrossDimX = (params.width + kChunkDim - 1) / kChunkDim;
    const int numChunksAcrossDimY = (params.height + kChunkDim - 1) / kChunkDim;

    // x_ptr holds the prefixes of the regular scan (conducted in forward pass).
    // For backward pass, it's used to:
    // (1) Restore the forward scan results;
    // (2) Reused as the prefixes for the reverse scan.
    // x_ptr (of dtype scan_t, used as prefixes for regular scans) has shape
    //                    batch_size,
    //                    dim,
    //                    numChunksDimY,
    //                    numChunksDimX,
    //                    dstate,
    //                    2,  // horizontal, vertical
    //                    kChunkDim,
    // x_ptr does NOT have the following extra
    //                   "2   // A, Bx" (as in Tensor x),
    // because it's scan_t, each element self-contains A and Bx!

    if (params.x_ptr == nullptr)
    {
        // x_ptr will be reused in the reverse scan; we want to ensure that it's not nullptr.
        printf("CUDA Kernel - InvalidArgument: x_ptr must not be nullptr");
        __trap();
    }

    scan_t * x_ptr =
            (params.x_ptr == nullptr)
            ? nullptr
            : reinterpret_cast<scan_t *>(params.x_ptr)
              + (batch_id * params.dim + dim_id)
                * (numChunksAcrossDimY * numChunksAcrossDimX * 2 * kChunkDim * params.dstate);

    auto x = [
            x_ptr, dstate = params.dstate,
            kChunkDim, kThreadSpan,
            numChunksAcrossDimX, numChunksAcrossDimY](
            int chunk_y_idx,
            int chunk_x_idx,
            int state_idx,
            ndmamba::ScanDir scanDir) -> scan_t *
    {
        int offset =
                chunk_y_idx * (numChunksAcrossDimX * dstate * 2 * kChunkDim)
                + chunk_x_idx * (dstate * 2 * kChunkDim)
                + state_idx * (2 * kChunkDim);

        // For regular scan, x_ptr + offset now points to a chunk of layout [
        //     [ kChunkDim elements for last col in this chunk (for horizontal scan) ]
        //     [ kChunkDim elements for last row in this chunk (for vertical scan) ]
        // ]

        // Or for reverse scan: x_ptr + offset now points to a chunk of layout [
        //     [ kChunkDim elements for first col in this chunk (for reverse horizontal scan) ]
        //     [ kChunkDim elements for first row in this chunk (for reverse vertical scan) ]
        // ]

        // Further offset it depending on whether we are doing horizontal or vertical scan.
        // For regular scans, x is already precomputed by forward pass and we just read the saved results.
        // For reverse scans, x is both read and written.

        switch (scanDir)
        {
            case ndmamba::kHorizontal:
            case ndmamba::kHorizontalReversed:
            {
                // threadIdx.x == 0, each threadIdx.y reads kThreadSpan elements in this column.
                offset += threadIdx.y * kThreadSpan;
                break;
            }
            case ndmamba::kVertical:
            case ndmamba::kVerticalReversed:
            {
                // threadIdx.y == 0, each threadIdx.x reads kThreadSpan elements in this row.
                offset += kChunkDim + threadIdx.x * kThreadSpan;
                break;
            }
            default:
            {
                printf("CUDA Kernel - InvalidArgument: unsupported scanDir %d in lambda x", scanDir);
                __trap();
            }
        }

        return x_ptr + offset;
    };

    // hbm_delta_a_exp saves the first row and column of delta_a_exp
    // for each batch_size, embed_dim, chunk_y_idx, chunk_x_idx, dstate).
    // The reversed scan needs SHIFTED delta_a_exp (right or bottom for rev hor and rev ver scans.)

    // hbm_delta_a_exp_ptr (of dtype float, used as prefixes for reverse scans) has shape
    //                        batch_size,
    //                        dim,
    //                        numChunksDimY,
    //                        numChunksDimX,
    //                        dstate,
    //                        2,  // horizontal, vertical
    //                        kChunkDim,
    // Note that kChunkDim might be greater than kBlockDim * KThreadSpan.

    // Each chunk will write to hbm_delta_a_exp_ptr (with its first column and first row).
    // Succeeding chunks will read corresponding locations in hbm_delta_a_exp_ptr to get their last columns and last rows
    // (which are shifted into different chunks, and thus must be read inter-chunk-ly from HBM).

    auto hbm_delta_a_exp_ptr = reinterpret_cast<float *>(params.rev_shift_tmp_ptr)
                               + (batch_id * params.dim + dim_id)
                                 * (numChunksAcrossDimY * numChunksAcrossDimX * 2 * kChunkDim * params.dstate);

    auto hbm_delta_a_exp = [
            hbm_delta_a_exp_ptr, dstate = params.dstate,
            kChunkDim, kThreadSpan,
            numChunksAcrossDimX, numChunksAcrossDimY](
            int chunk_y_idx,
            int chunk_x_idx,
            int state_idx,
            ndmamba::ScanDir scanDir) -> float *
    {
        int offset =
                chunk_y_idx * (numChunksAcrossDimX * dstate * 2 * kChunkDim)
                + chunk_x_idx * (dstate * 2 * kChunkDim)
                + state_idx * (2 * kChunkDim);

        // Or for reverse scan: hbm_delta_a_exp_ptr + offset now points to a chunk of layout [
        //     [ kChunkDim elements for first col in this chunk's delta_a_exp ]
        //     [ kChunkDim elements for first row in this chunk's delta_a_exp ]
        // ]

        // Further offset it depending on whether we are doing horizontal or vertical scan.

        switch (scanDir)
        {
            case ndmamba::kHorizontalReversed:
            {
                // threadIdx.x == 0, each threadIdx.y reads kThreadSpan elements in this column.
                offset += threadIdx.y * kThreadSpan;
                break;
            }
            case ndmamba::kVerticalReversed:
            {
                // threadIdx.y == 0, each threadIdx.x reads kThreadSpan elements in this row.
                offset += kChunkDim + threadIdx.x * kThreadSpan;
                break;
            }
            default:
            {
                printf("CUDA Kernel - InvalidArgument: unsupported scanDir %d in lambda hbm_delta_a_exp", scanDir);
                __trap();
            }
        }

        return hbm_delta_a_exp_ptr + offset;
    };

    #if false
    // Offset to first element in last chunk
    constexpr int kChunkSize = kNThreads * kNItems;
    u += (params.n_chunks - 1) * kChunkSize;
    delta += (params.n_chunks - 1) * kChunkSize;
    dout += (params.n_chunks - 1) * kChunkSize;
    Bvar += (params.n_chunks - 1) * kChunkSize * (!kIsComplex ? 1 : 2);
    Cvar += (params.n_chunks - 1) * kChunkSize * (!kIsComplex ? 1 : 2);
    #endif  // false

    float dD_val = 0;
    float ddelta_bias_val = 0;

    for (int chunk_y_idx = numChunksAcrossDimY - 1; 0 <= chunk_y_idx; --chunk_y_idx)
    {
        for (int chunk_x_idx = numChunksAcrossDimX - 1; 0 <= chunk_x_idx; --chunk_x_idx)
        {
            input_t u_vals[kThreadSpan][kThreadSpan];
//             input_t delta_vals_load[kThreadSpan][kThreadSpan];
//             input_t dout_vals_load[kThreadSpan][kThreadSpan];
            float dout_vals[kThreadSpan][kThreadSpan];
            float delta_vals[kThreadSpan][kThreadSpan];

            const int thread_x = kChunkDim * chunk_x_idx + threadIdx.x * kThreadSpan;
            const int thread_y = kChunkDim * chunk_y_idx + threadIdx.y * kThreadSpan;

            // Load input (u, delta) and dOut
            #pragma unroll
            for (int yy = 0; yy < kThreadSpan; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < kThreadSpan; ++xx)
                {
                    const int gx = thread_x + xx;
                    const int gy = thread_y + yy;

                    if (gx < params.width && gy < params.height)
                    {
                        u_vals[yy][xx] = u[gy * params.width + gx];
//                         delta_vals_load[yy][xx] = delta[gy * params.width + gx];
//                         dout_vals_load[yy][xx] = static_cast<float>(dout[gy * params.width + gx]);
                        dout_vals[yy][xx] = static_cast<float>(dout[gy * params.width + gx]);
                        delta_vals[yy][xx] = static_cast<float>(delta[gy * params.width + gx]) + delta_bias;

                        #ifdef NAN_GRAD_CHECK
                        if (isnan(dout_vals[yy][xx]))
                        {
                            printf("CUDA Kernel - RuntimeError: "
                                   "NaN dout_vals detected at "
                                   "selective_scan/selective_sacn_bwd_kernel.cuh:536!"
                                   "block (%d %d), thread(%d %d)\n",
                                   blockIdx.x, blockIdx.y,
                                   threadIdx.x, threadIdx.y);
                            __trap();
                        }
                        #endif  // NAN_GRAD_CHECK

                    }
                    else
                    {
                        u_vals[yy][xx] = 0;
//                         delta_vals_load[yy][xx] = 1.0;
//                         dout_vals_load[yy][xx] = 0;
                        delta_vals[yy][xx] = 1;
                        dout_vals[yy][xx] = 0;
                    }

                    if constexpr (kDeltaSoftplus)
                    {
                        delta_vals[yy][xx] =
                                delta_vals[yy][xx] <= 20.0f ?
                                log1pf(expf(delta_vals[yy][xx])) :
                                delta_vals[yy][xx];
                    }
                }
            }

//             #pragma unroll
//             for (int yy = 0; yy < kThreadSpan; ++yy)
//             {
//                 #pragma unroll
//                 for (int xx = 0; xx < kThreadSpan; ++xx)
//                 {
//                     dout_vals[yy][xx] = static_cast<float>(dout_vals_load[yy][xx]);
//                     delta_vals[yy][xx] = static_cast<float>(delta_vals_load[yy][xx]) + delta_bias;
//
//                     if constexpr (kDeltaSoftplus)
//                     {
//                         delta_vals[yy][xx] =
//                                 delta_vals[yy][xx] <= 20.0f ?
//                                 log1pf(expf(delta_vals[yy][xx])) :
//                                 delta_vals[yy][xx];
//                     }
//                 }
//             }

            if constexpr (kHasZ)
            {
                input_t * z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                              + dim_id * params.z_d_stride;
                output_t * out = reinterpret_cast<output_t *>(params.out_ptr) + batch_id * params.out_batch_stride
                                + dim_id * params.out_d_stride;
                input_t * dz = reinterpret_cast<input_t *>(params.dz_ptr) + batch_id * params.dz_batch_stride
                               + dim_id * params.dz_d_stride;

                // Load inputs
                output_t z_vals[kThreadSpan][kThreadSpan];
                output_t out_vals[kThreadSpan][kThreadSpan];

                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        const int gx = thread_x + xx;
                        const int gy = thread_y + yy;

                        if (gx < params.width && gy < params.height)
                        {
                            z_vals[yy][xx] = z[gy * params.width + gx];
                            out_vals[yy][xx] = out[gy * params.width + gx];
                        }
                        else
                        {
                            z_vals[yy][xx] = 0;
                            out_vals[yy][xx] = 0;
                        }
                    }
                }

                // Compute gradients
                float dz_vals[kThreadSpan][kThreadSpan];
                float z_silu_vals[kThreadSpan][kThreadSpan];

                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        float z_val = z_vals[yy][xx];
                        float z_sigmoid_val = 1.0f / (1.0f + expf(-z_val));

                        z_silu_vals[yy][xx] = z_val * z_sigmoid_val;

                        dz_vals[yy][xx] =
                                dout_vals[yy][xx] * static_cast<float>(out_vals[yy][xx]) *
                                z_sigmoid_val * (1.0f + z_val * (1.0f - z_sigmoid_val));

                        dout_vals[yy][xx] *= z_silu_vals[yy][xx];
                    }
                }

                // Store gradients
                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        const int gx = thread_x + xx;
                        const int gy = thread_y + yy;

                        if (gx < params.width && gy < params.height)
                        {
                            dz[gy * params.width + gx] = static_cast<input_t>(dz_vals[yy][xx]);
                        }
                    }
                }

                // ndmamba: NOT used.
                #if false
                if (params.out_z_ptr != nullptr)
                {
                    // Recompute and store out_z
                    float out_z_vals[kNItems];
                    #pragma unroll
                    for (int i = 0; i < kNItems; ++i)
                    {
                        out_z_vals[i] = float(out_vals[i]) * z_silu_vals[i];
                    }
                    // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
                    // printf("out_val=%f, z_silu_val = %f, out_z_val = %f\n", float(out_vals[0]), z_silu_vals[0], out_z_vals[0]);
                    // }
                    input_t * out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                                      + dim_id * params.out_z_d_stride + chunk * kChunkSize;
                    __syncthreads();
                    store_output<Kernel>(out_z, out_z_vals, smem_store, params.seqlen - chunk * kChunkSize);
                }
                #endif  // false
            }

            float du_vals[kThreadSpan][kThreadSpan];

            #pragma unroll
            for (int yy = 0; yy < kThreadSpan; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < kThreadSpan; ++xx)
                {
                    du_vals[yy][xx] = D_val * dout_vals[yy][xx];
                }
            }

            #pragma unroll
            for (int yy = 0; yy < kThreadSpan; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < kThreadSpan; ++xx)
                {
                    // ndmamba: dD is same across L, will be block-reduced.
                    dD_val += dout_vals[yy][xx] * static_cast<float>(u_vals[yy][xx]);
                }
            }

            float ddelta_vals[kThreadSpan][kThreadSpan] = {0};

            for (int state_idx = 0; state_idx < params.dstate; ++state_idx)
            {
                __syncthreads();

                // read A, A: D, N
                const weight_t A_val = A[state_idx * params.A_dstate_stride];

                // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
                weight_t A_scaled;
                constexpr float kLog2e = M_LOG2E;

                if constexpr (!kIsComplex)
                {
                    A_scaled = A_val * kLog2e;
                }
                else
                {
                    A_scaled = complex_t(A_val.real_ * kLog2e, A_val.imag_);
                }

                // B_val and C_val holds B and C if B or C are constant across seqlen.
                // If B or C varies, B_val or C_val is unused.
                weight_t B_val;
                weight_t C_val;
                weight_t B_vals[kThreadSpan][kThreadSpan];
                weight_t C_vals[kThreadSpan][kThreadSpan];

                if constexpr (!kIsVariableB)
                {
                    B_val = B[state_idx * params.B_dstate_stride];
                }
                else
                {
                    #pragma unroll
                    for (int yy = 0; yy < kThreadSpan; ++yy)
                    {
                        #pragma unroll
                        for (int xx = 0; xx < kThreadSpan; ++xx)
                        {
                            const int gx = thread_x + xx;
                            const int gy = thread_y + yy;

                            if (gx < params.width && gy < params.height)
                            {
                                B_vals[yy][xx] = Bvar[state_idx * params.B_dstate_stride + gy * params.width + gx];
                            }
                            else
                            {
                                if constexpr (Kernel::kIsComplex)
                                {
                                    B_vals[yy][xx] = complex_t(0.0f, 0.0f);
                                }
                                else
                                {
                                    B_vals[yy][xx] = 0.0f;
                                }
                            }
                        }
                    }
                }

                if constexpr (!kIsVariableC)
                {
                    C_val = C[state_idx * params.C_dstate_stride];
                }
                else
                {
                    #pragma unroll
                    for (int yy = 0; yy < kThreadSpan; ++yy)
                    {
                        #pragma unroll
                        for (int xx = 0; xx < kThreadSpan; ++xx)
                        {
                            const int gx = thread_x + xx;
                            const int gy = thread_y + yy;

                            if (gx < params.width && gy < params.height)
                            {
                                C_vals[yy][xx] = Cvar[state_idx * params.C_dstate_stride + gy * params.width + gx];
                            }
                            else
                            {
                                if constexpr (Kernel::kIsComplex)
                                {
                                    C_vals[yy][xx] = complex_t(0.0f, 0.0f);
                                }
                                else
                                {
                                    C_vals[yy][xx] = 0.0f;
                                }
                            }
                        }
                    }
                }

                // Prepare scan data. Load from HBM.
                __syncthreads();
//                 scan_t thread_data[kThreadSpan][kThreadSpan];
                scan_t thread_reverse_data[kThreadSpan][kThreadSpan];

                #ifdef NAN_SMEM_CHECK_INIT_WITH_NAN
                // For debugging:
                // Manually fill all cache elements with nan
                // so uninitialized elements will immediately result in nan loss.
                if (threadIdx.x == 0 && threadIdx.y == 0)
                {
                    for (int cay = 0; cay < kLdSmemDeltaAExp; ++cay)
                    {
                        for (int cax = 0; cax < kLdSmemDeltaAExp; ++cax)
                        {
                            smem_delta_a_exp[cay * kLdSmemDeltaAExp + cax] = CUDART_NAN_F;
                        }
                    }
                }
                __syncthreads();
                #endif  // NAN_SMEM_CHECK_INIT_WITH_NAN

                // delta_a_exp is cached in SMEM for shifting.
                // This smem is of shape (kChunkDim + 1, kChunkDim + 1,).
                // The top-left submatrix holds delta_a_exp of the current chunk.
                // The rightmost column and bottommost row hold result of the previous chunk.
                // Load the boundary (last col and last row)
                // either from HBM (written by previous chunks, when there is a previous chunk in this direction),
                // or with trivial value ((1, 0) for (A, Bu)).

                // chunk_x_idx + 1, and chunk_x_idx + 1 might go out-of-bound,
                // but it's OK becuase this is just offseting a pointer.
                // The out-of-bound pointers will not be dereferenced by their hosting threads.
                float * hbm_delta_a_exp_first_col_of_previous_chunk = hbm_delta_a_exp(
                        chunk_y_idx, chunk_x_idx + 1, state_idx, ndmamba::kHorizontalReversed);
                float * hbm_delta_a_exp_first_row_of_previous_chunk = hbm_delta_a_exp(
                        chunk_y_idx + 1, chunk_x_idx, state_idx, ndmamba::kVerticalReversed);

                // Load last column (without lower-right corner) of delta_a_exp by threads in last column of block.
                // This is for use of reverse horizontal scan, trivial value when chunk_x_idx == numChunksAcrossDimX - 1.
                if (threadIdx.x == kBlockDim - 1)
                {
                    const int sx = kLdSmemDeltaAExp - 1;

                    #pragma unroll
                    for (int yy = 0; yy < kThreadSpan; ++yy)
                    {
                        const int sy = threadIdx.y * kThreadSpan + yy;

                        smem_delta_a_exp[sy * kLdSmemDeltaAExp + sx] =
                                chunk_x_idx == numChunksAcrossDimX - 1
                                ? 1.0f
                                : hbm_delta_a_exp_first_col_of_previous_chunk[yy];

                        #ifdef CHECK_DELTA_A_EXP_PREFIXES
                        if (chunk_y_idx == 1 && chunk_x_idx == 0)
                            printf("LOADED DELTA_A_EXP COL chunk (y=%d x=%d) "
                                   "loc (y=%d x=%d) "
//                                   "hbm ptr @ %p = %p + %d "
                                   "delta_a_exp = %f\n",
                                   chunk_y_idx, chunk_x_idx,
                                   sy, 0,
//                                   hbm_delta_a_exp_first_col_of_previous_chunk + yy,
//                                   hbm_delta_a_exp_first_col_of_previous_chunk, yy,
                                   smem_delta_a_exp[sy * kLdSmemDeltaAExp + sx]);
                        #endif  // CHECK_DELTA_A_EXP_PREFIXES
                    }
                }

                // Load last row (without lower-right corner) of delta_a_exp by threads in last row of block
                if (threadIdx.y == kBlockDim - 1)
                {
                    const int sy = kLdSmemDeltaAExp - 1;


                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        const int sx = threadIdx.x * kThreadSpan + xx;

                        smem_delta_a_exp[sy * kLdSmemDeltaAExp + sx] =
                                chunk_y_idx == numChunksAcrossDimY - 1
                                ? 1.0f
                                : hbm_delta_a_exp_first_row_of_previous_chunk[xx];

//                        if (abs(smem_delta_a_exp[sy * kLdSmemDeltaAExp + sx] - 2333.6666f) > 1e-4f)
//                        {
//                            printf("chunk (y=%d x=%d) "
//                                   "thread (y=%d x=%d) "
//                                   "read uninitialized data [%d]"
//                                   "numChunks (y=%d, x=%d)\n",
//                                   chunk_y_idx, chunk_x_idx,
//                                   threadIdx.y, threadIdx.x,
//                                   xx,
//                                   numChunksAcrossDimY, numChunksAcrossDimX);
//                            __trap();
//                        }

                        #ifdef CHECK_DELTA_A_EXP_PREFIXES
//                        if (chunk_y_idx == 1 && chunk_x_idx == 0)
//                        printf("LOADED DELTA_A_EXP ROW chunk (y=%d x=%d) "
//                               "loc (y=%d x=%d) "
////                               "hbm ptr @ %p = %p + %d "
//                               "delta_a_exp = %f\n",
//                               chunk_y_idx, chunk_x_idx,
//                               0, sx,
////                               hbm_delta_a_exp_first_col_of_previous_chunk + xx,
////                               hbm_delta_a_exp_first_col_of_previous_chunk, xx,
//                               smem_delta_a_exp[sy * kLdSmemDeltaAExp + sx]);
                        #endif  // CHECK_DELTA_A_EXP_PREFIXES
                    }
                }

                // Load the lower-right corner of delta_a_exp.
                // TODO(Xi): This location should not be accessed by any thread, maybe we don't need to assign it here?
                if (threadIdx.x == 0 && threadIdx.y == 0)
                {
                    smem_delta_a_exp[kLdSmemDeltaAExp * kLdSmemDeltaAExp - 1] = CUDART_NAN_F;
                }

                // No __syncthreads() needed here
                // as we'll be loading more values into smem_delta_a_exp (aliased by deltaAExpCache below).

                // delta_a_exp computaion.
                // (1) Calculate delta_a_exp at each location and save to SMEM.
                // (2) Save the first column and the first row into HBM for future use by succeeding chunks.
                float * delta_a_exp_first_column_for_next_chunk = hbm_delta_a_exp(
                        chunk_y_idx, chunk_x_idx, state_idx, ndmamba::kHorizontalReversed);
                float * delta_a_exp_first_row_for_next_chunk = hbm_delta_a_exp(
                        chunk_y_idx, chunk_x_idx, state_idx, ndmamba::kVerticalReversed);

                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        const int gx = thread_x + xx;
                        const int gy = thread_y + yy;

                        const float delta_a_exp =
                                (gx < params.width && gy < params.height)
                                ? exp2f(delta_vals[yy][xx] * A_scaled)
                                : 1.0f;

                        // Not all elements in smem_delta_a_exp are properly set here.
                        // The last col and last row are left uninitlized!
                        // These must be manually set to previous states before flushing new values.
                        deltaAExpCache(yy, xx) = delta_a_exp;

                        // Save first column to HBM.
                        // First-col threads' first-colmun data
                        if (0 == xx && 0 == threadIdx.x)
                        {
                            delta_a_exp_first_column_for_next_chunk[yy] = delta_a_exp;

                            #ifdef CHECK_DELTA_A_EXP_PREFIXES
                            if (chunk_y_idx == 1 && chunk_x_idx == 1)
                            printf("SAVED DELTA_A_EXP COL chunk (y=%d x=%d) "
                                   "loc (y=%d x=%d) "
//                                   "hbm ptr @ %p = %p + %d "
                                   "delta_a_exp = %f\n",
                                   chunk_y_idx, chunk_x_idx,
                                   threadIdx.y * kThreadSpan + yy, 0,
//                                   delta_a_exp_first_column_for_next_chunk + yy,
//                                   delta_a_exp_first_column_for_next_chunk, yy,
                                   delta_a_exp_first_column_for_next_chunk[yy]);
                            #endif  // CHECK_DELTA_A_EXP_PREFIXES
                        }

                        // Save first row to HBM.
                        // First-row threads' first-row data
                        if (0 == yy && 0 == threadIdx.y)
                        {
                            delta_a_exp_first_row_for_next_chunk[xx] = delta_a_exp;

//                            if (chunk_y_idx == 1 && chunk_x_idx == 1)
//                            printf("SAVED DELTA_A_EXP ROW chunk (y=%d x=%d) "
//                                   "loc (y=%d x=%d) "
////                                   "hbm ptr @ %p = %p + %d "
//                                   "delta_a_exp = %f\n",
//                                   chunk_y_idx, chunk_x_idx,
//                                   0, threadIdx.x * kThreadSpan + xx,
////                                   delta_a_exp_first_row_for_next_chunk + xx,
////                                   delta_a_exp_first_row_for_next_chunk, xx,
//                                   delta_a_exp_first_row_for_next_chunk[xx]);
                        }

                        #ifdef NAN_SMEM_CHECK
                        if (isnan(delta_a_exp))
                        {
                            printf("CUDA Kernel - RuntimeError: "
                                   "NaN delta_a_exp computed at "
                                   "selective_scan/selective_sacn_bwd_kernel.cuh:711! "
                                   "block (%d %d) thread (%d %d)\n",
                                   blockIdx.x, blockIdx.y,
                                   threadIdx.x, threadIdx.y);
                            __trap();
                        }
                        #endif  // NAN_SMEM_CHECK
                    }
                }

                #ifdef NAN_SMEM_CHECK
                if (threadIdx.x == 0 && threadIdx.y == 0)
                {
                    for (int cay = 0; cay < kLdSmemDeltaAExp; ++cay)
                    {
                        for (int cax = 0; cax < kLdSmemDeltaAExp; ++cax)
                        {
                            if (isnan(smem_delta_a_exp[cay * kLdSmemDeltaAExp + cax]))
                            {
                                printf("CUDA Kernel - RuntimeError: "
                                       "NaN smem_delta_a_exp detected at "
                                       "selective_scan/selective_sacn_bwd_kernel.cuh:729!\n");
                                __trap();
                            }
                        }
                    }
                }
                __syncthreads();
                #endif  // NAN_SMEM_CHECK

                // MUST SYNC after mofidying SMEM!!!
                __syncthreads();

                ///
                /// REGULAR SCAN
                ///

                // Set data-to-scan.
                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        // ndmamba: Load delta_a_exp (previously cached in SMEM).

                        thread_reverse_data[yy][xx] = make_float2(
                                deltaAExpCache(yy, xx),
                                !kIsVariableB
                                ? delta_vals[yy][xx] * float(u_vals[yy][xx])
                                : delta_vals[yy][xx] * float(u_vals[yy][xx]) * B_vals[yy][xx]
                        );

                        #ifdef NAN_GRAD_CHECK
                        if (isnan(thread_reverse_data[yy][xx].x))
                        {
                            printf("CUDA Kernel - RuntimeError: "
                                   "NaN thread_reverse_data.x detected at "
                                   "selective_scan/selective_sacn_bwd_kernel.cuh:790 "
                                   "(before horizontal scan)!"
                                   "block (%d %d) thread (%d %d)\n",
                                   blockIdx.x, blockIdx.y,
                                   threadIdx.x, threadIdx.y);
                            __trap();
                        }
                        if (isnan(thread_reverse_data[yy][xx].y))
                        {
                            printf("CUDA Kernel - RuntimeError: "
                                   "NaN thread_reverse_data.y detected at "
                                   "selective_scan/selective_sacn_bwd_kernel.cuh:801 "
                                   "(before horizontal scan)!"
                                   "block (%d %d) thread (%d %d)\n",
                                   blockIdx.x, blockIdx.y,
                                   threadIdx.x, threadIdx.y);
                            __trap();
                        }
                        #endif  // NAN_GRAD_CHECK
                    }
                }

                __syncthreads();

                // Load regular scan prefixes (needed when there are multiple chunks)
                SSMScanPrefixCallbackOp<weight_t> prefix_ops_for_horizontal_scan[kThreadSpan];
                SSMScanPrefixCallbackOp<weight_t> prefix_ops_for_vertical_scan[kThreadSpan];

                scan_t * prefixes_for_horizontal_scan = x(
                        chunk_y_idx, chunk_x_idx - 1, state_idx, ndmamba::kHorizontal);
                scan_t * prefixes_for_vertical_scan = x(
                        chunk_y_idx - 1, chunk_x_idx, state_idx, ndmamba::kVertical);

                #pragma unroll
                for (int yx = 0; yx < kThreadSpan; ++yx)
                {
                    prefix_ops_for_horizontal_scan[yx].running_prefix =
                            (0 < chunk_x_idx && threadIdx.x == 0)
                            ? prefixes_for_horizontal_scan[yx]
                            : make_float2(1.0f, 0.0f);

                    prefix_ops_for_vertical_scan[yx].running_prefix =
                            (0 < chunk_y_idx && threadIdx.y == 0)
                            ? prefixes_for_vertical_scan[yx]
                            : make_float2(1.0f, 0.0f);
                }

                typename Kernel::BlockScan scan(scanTempStorage);

                // Restore h_s
                scan.InclusiveScan(
                        thread_reverse_data,
                        thread_reverse_data,
                        SSMScanOp<weight_t>(),
                        prefix_ops_for_horizontal_scan,
                        ndmamba::kHorizontal
                );

                // Save results for h_s.
                float horizontal_results[kThreadSpan][kThreadSpan];

                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        horizontal_results[yy][xx] = thread_reverse_data[yy][xx].y;
                    }
                }

                // Re-initialize A (aka thread_reverse_data.x) (used as thread_data)
                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        // No boundary test needed here
                        // as the cache automatically stores ONE
                        // if out of bound.
                        thread_reverse_data[yy][xx].x = deltaAExpCache(yy, xx);

                        #ifdef NAN_GRAD_CHECK
                        if (isnan(thread_reverse_data[yy][xx].x))
                        {
                            printf("CUDA Kernel - RuntimeError: "
                                   "NaN thread_reverse_data.x detected at "
                                   "selective_scan/selective_sacn_bwd_kernel.cuh:919 "
                                   "(after horizontal scan, before vertical scan)!"
                                   "block (%d %d) thread (%d %d)\n",
                                   blockIdx.x, blockIdx.y,
                                   threadIdx.x, threadIdx.y);
                            __trap();
                        }
                        if (isnan(thread_reverse_data[yy][xx].y))
                        {
                            printf("CUDA Kernel - RuntimeError: "
                                   "NaN thread_reverse_data.y detected at "
                                   "selective_scan/selective_sacn_bwd_kernel.cuh:930 "
                                   "(after horizontal scan, before vertical scan)!"
                                   "block (%d %d) thread (%d %d)\n",
                                   blockIdx.x, blockIdx.y,
                                   threadIdx.x, threadIdx.y);
                            __trap();
                        }
                        #endif  // NAN_GRAD_CHECK
                    }
                }

                // Restore h_v
                scan.InclusiveScan(
                        thread_reverse_data,
                        thread_reverse_data,
                        SSMScanOp<weight_t>(),
                        prefix_ops_for_vertical_scan,
                        ndmamba::kVertical
                );

                // Save results for h_s.
                // Meanwhile, set values for reverse scan.
                float vertical_results[kThreadSpan][kThreadSpan];
                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        vertical_results[yy][xx] = thread_reverse_data[yy][xx].y;

                        thread_reverse_data[yy][xx].x = deltaAExpCacheShifted(yy, xx, ndmamba::kVerticalReversed);

                        thread_reverse_data[yy][xx].y = dout_vals[yy][xx] * (
                                !kIsVariableC
                                ? (!kIsVariableB ? B_val * C_val : C_val)
                                : (!kIsVariableB ? B_val * C_vals[yy][xx] : C_vals[yy][xx])
                        );

                        #ifdef NAN_GRAD_CHECK
                        if (isnan(thread_reverse_data[yy][xx].x))
                        {
                            printf("CUDA Kernel - RuntimeError: "
                                   "NaN thread_reverse_data.x detected at "
                                   "selective_scan/selective_sacn_bwd_kernel.cuh:976 "
                                   "(after vertical scan, before vertical reverse scan)!"
                                   "block (%d %d) thread (%d %d)\n",
                                   blockIdx.x, blockIdx.y,
                                   threadIdx.x, threadIdx.y);
                            __trap();
                        }
                        if (isnan(thread_reverse_data[yy][xx].y))
                        {
                            printf("CUDA Kernel RuntimeError: "
                                   "NaN thread_reverse_data.y detected at "
                                   "selective_scan/selective_sacn_bwd_kernel.cuh:987 "
                                   "(after vertical scan, before vertical reverse scan)!"
                                   "block (%d %d) thread (%d %d)\n",
                                   blockIdx.x, blockIdx.y,
                                   threadIdx.x, threadIdx.y);
                            __trap();
                        }
                        #endif  // NAN_GRAD_CHECK
                    }
                }

                ///
                /// REVERSE SCAN FROM HERE
                ///

                // Load prefixes for reverse scans.
                // Reuse the pointers and prefix ops to save registers.
                // Also reuse the HBM space of x_ptr.
                prefixes_for_horizontal_scan = x(
                        chunk_y_idx, chunk_x_idx + 1, state_idx, ndmamba::kHorizontalReversed);
                prefixes_for_vertical_scan = x(
                        chunk_y_idx + 1, chunk_x_idx, state_idx, ndmamba::kVerticalReversed);

                #pragma unroll
                for (int yx = 0; yx < kThreadSpan; ++yx)
                {
                    // Threads handling the last column in reverse horizontal scan should load the prefix.
                    // Such threads have threadIdx.x == kBlockDim - 1.
                    prefix_ops_for_horizontal_scan[yx].running_prefix =
                            (chunk_x_idx < numChunksAcrossDimX - 1 && threadIdx.x == kBlockDim - 1)
                            ? prefixes_for_horizontal_scan[yx]
                            : make_float2(1.0f, 0.0f);

                    // Threads handling the last row in reverse vertical scan should load the prefix.
                    // Such threads have threadIdx.y == kBlockDim - 1.
                    prefix_ops_for_vertical_scan[yx].running_prefix =
                            (chunk_y_idx < numChunksAcrossDimY - 1 && threadIdx.y == kBlockDim - 1)
                            ? prefixes_for_vertical_scan[yx]
                            : make_float2(1.0f, 0.0f);


                    if (chunk_y_idx == 1 && chunk_x_idx == 0)
                    {
                        #ifdef CHECK_SCAN_PREFIXES
                        if (threadIdx.x == kBlockDim - 1)
                        printf("LOADED HORI PREFIX chunk (y=%d x=%d) "
                               "row %d "
                               "postfix = %4.3f, %4.3f\n",
                               chunk_y_idx, chunk_x_idx,
                               threadIdx.y * kThreadSpan + yx,
                               prefix_ops_for_horizontal_scan[yx].running_prefix.x,
                               prefix_ops_for_horizontal_scan[yx].running_prefix.y
                        );

//                        if (threadIdx.y == kBlockDim - 1)
//                        printf("LOADED VERT PREFIX chunk (y=%d x=%d) "
//                               "col %d "
//                               "postfix = %4.3f, %4.3f\n",
//                               chunk_y_idx, chunk_x_idx,
//                               threadIdx.x * kThreadSpan + yx,
//                               prefix_ops_for_vertical_scan[yx].running_prefix.x,
//                               prefix_ops_for_vertical_scan[yx].running_prefix.y
//                        );
                        #endif  // CHECK_SCAN_PREFIXES
                    }
                }

                scan.InclusiveScan(
                        thread_reverse_data,
                        thread_reverse_data,
                        SSMScanOp<weight_t>(),
                        prefix_ops_for_vertical_scan,
                        ndmamba::kVerticalReversed
                );

                // "SHIFT" THREAD_DATA.Y AND MULTIPLY WITH REV VERT SCAN RES, STORE INTO dA.
                // Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])
                // Q is dA, X is thread_data.y, grad_output is thread_reverse_data.y.
                weight_t dA_val = 0;

                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        const float dx = thread_reverse_data[yy][xx].y;
                        const float a = vertical_results[yy][xx] - horizontal_results[yy][xx];
                        dA_val += dx * delta_vals[yy][xx] * a;

                        // This line of code assumes delta is not softplus-ed.
                        // When there's softplus, ddelta is recomputed.
                        // Only considers ddelta's components over exp(deltaA).
                        // Component over deltaBu is accumulated afterward.
                        ddelta_vals[yy][xx] += dx * A_val * a;
                    }
                }

                // Re-initialize A (aka thread_data.x)
                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        // No boundary test needed here
                        // as the cache automatically stores ONE
                        // if out of bound.
                        thread_reverse_data[yy][xx].x = deltaAExpCacheShifted(yy, xx, ndmamba::kHorizontalReversed);

                        #ifdef NAN_GRAD_CHECK
                        if (isnan(thread_reverse_data[yy][xx].x))
                        {
                            printf("CUDA Kernel - RuntimeError: "
                                   "NaN thread_reverse_data.x detected at "
                                   "selective_scan/selective_sacn_bwd_kernel.cuh:1070 "
                                   "(after vertical reverse scan, before horizontal reverse scan)!"
                                   "block (%d %d) thread (%d %d)\n",
                                   blockIdx.x, blockIdx.y,
                                   threadIdx.x, threadIdx.y);
                            __trap();
                        }
                        if (isnan(thread_reverse_data[yy][xx].y))
                        {
                            printf("CUDA Kernel - RuntimeError: "
                                   "NaN thread_reverse_data.y detected at "
                                   "selective_scan/selective_sacn_bwd_kernel.cuh:1081 "
                                   "(after vertical reverse scan, before horizontal reverse scan)!"
                                   "block (%d %d) thread (%d %d)\n",
                                   blockIdx.x, blockIdx.y,
                                   threadIdx.x, threadIdx.y);
                            __trap();
                        }
                        #endif  // NAN_GRAD_CHECK
                    }
                }

                scan.InclusiveScan(
                        thread_reverse_data,
                        thread_reverse_data,
                        SSMScanOp<weight_t>(),
                        prefix_ops_for_horizontal_scan,
                        ndmamba::kHorizontalReversed
                );

                // Write the running postfixes (but named prefixes here) of this chunk
                // to HBM for further use by neighboring chunks.
                prefixes_for_horizontal_scan = x(chunk_y_idx, chunk_x_idx, state_idx, ndmamba::kHorizontalReversed);
                prefixes_for_vertical_scan = x(chunk_y_idx, chunk_x_idx, state_idx, ndmamba::kVerticalReversed);

                #pragma unroll
                for (int yx = 0; yx < kThreadSpan; ++yx)
                {
                    if (threadIdx.x == kBlockDim - 1)
                    {
                        prefixes_for_horizontal_scan[yx] = prefix_ops_for_horizontal_scan[yx].running_prefix;

                        #ifdef CHECK_SCAN_PREFIXES
                        if (chunk_y_idx == 1 && chunk_x_idx == 1)
                        {
                            printf("SAVED HORI PREFIX chunk (y=%d x=%d) "
                                   "row %d "
                                   "postfix = %4.3f, %4.3f\n",
                                   chunk_y_idx, chunk_x_idx,
                                   threadIdx.y * kThreadSpan + yx,
                                   prefixes_for_horizontal_scan[yx].x,
                                   prefixes_for_horizontal_scan[yx].y
                            );
                        }
                        #endif  // CHECK_SCAN_PREFIXES
                    }

                    if (threadIdx.y == kBlockDim - 1)
                    {
                        prefixes_for_vertical_scan[yx] = prefix_ops_for_vertical_scan[yx].running_prefix;

//                        if (chunk_y_idx == 1 && chunk_x_idx == 1)
//                        {
//                            printf("SAVED VERT PREFIX chunk (y=%d x=%d) "
//                                   "col %d "
//                                   "postfix = %4.3f, %4.3f\n",
//                                   chunk_y_idx, chunk_x_idx,
//                                   threadIdx.x * kThreadSpan + yx,
//                                   prefixes_for_vertical_scan[yx].x,
//                                   prefixes_for_vertical_scan[yx].y
//                            );
//                        }
                    }
                }

                ///
                /// REMAINING GRAD CALCULATION
                ///

                // "SHIFT" HORIZONTAL_RESULT AND MULTIPLY WITH REV VERT SCAN RES, STORE INTO dA.
                // mamba.py says Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])
                // Here, Q is dA, X is thread_data.y, grad_output is thread_reverse_data.y.
                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        const float dx = thread_reverse_data[yy][xx].y;

                        const float a = horizontal_results[yy][xx] - (
                                !kIsVariableB
                                ? delta_vals[yy][xx] * static_cast<float>(u_vals[yy][xx])
                                : delta_vals[yy][xx] * static_cast<float>(u_vals[yy][xx]) * B_vals[yy][xx]
                        );

                        dA_val += dx * delta_vals[yy][xx] * a;

                        // This line of code assumes delta is not softplus-ed.
                        // When there's softplus, ddelta is recomputed.
                        ddelta_vals[yy][xx] += dx * A_val * a;
                    }
                }

                // ddelta and du.
                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        const float dx = thread_reverse_data[yy][xx].y;
                        const float ddelta_u = !kIsVariableB ? dx : dx * B_vals[yy][xx];

                        du_vals[yy][xx] += ddelta_u * delta_vals[yy][xx];
                        ddelta_vals[yy][xx] += ddelta_u * static_cast<float>(u_vals[yy][xx]);
                    }
                }

//                    if (threadIdx.x == 0)
//                    {
//                        smem_running_postfix[state_idx] = postfix_op.running_prefix;
//                    }

                // dB and dC.
                weight_t dBC_val = 0;
                weight_t dB_vals[kThreadSpan][kThreadSpan];
                weight_t dC_vals[kThreadSpan][kThreadSpan];

                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        if constexpr (!kIsVariableB || !kIsVariableC)
                        {
                            if constexpr (!kIsVariableB)
                            {
                                // B is const
                                // dBC_val is dB_val
                                dBC_val += dout_vals[yy][xx] * (
                                        !kIsVariableC
                                        ? vertical_results[yy][xx]
                                        : vertical_results[yy][xx] * C_vals[yy][xx]
                                );
                            }
                            else
                            {
                                // C is const
                                // dBC_val is dC_val
                                dBC_val += dout_vals[yy][xx] * vertical_results[yy][xx];
                            }
                        }

                        if constexpr (kIsVariableB)
                        {
                            const float dx = thread_reverse_data[yy][xx].y;
                            dB_vals[yy][xx] = dx * delta_vals[yy][xx] * static_cast<float>(u_vals[yy][xx]);
                        }

                        if constexpr (kIsVariableC)
                        {
                            dC_vals[yy][xx] = dout_vals[yy][xx] * (
                                    !kIsVariableB
                                    ? vertical_results[yy][xx] * B_val // B const
                                    : vertical_results[yy][xx]  // B var
                            );
                        }
                    }
                }

                // vanilla mamba: Block-exchange to make the atomicAdd's coalesced, otherwise they're much slower
                // ndmamba: use __syncthreads instead of atomicAdd
                if constexpr (kIsVariableB || kIsVariableC)
                {
//                        if constexpr (kIsVariableB)
//                        {
//                            typename Kernel::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals, dB_vals);
//                        }
//
//                        if constexpr (kIsVariableC)
//                        {
//                            auto & smem_exchange_C = !kIsVariableB ? smem_exchange : smem_exchange1;
//                            typename Kernel::BlockExchangeT(smem_exchange_C).BlockedToStriped(dC_vals, dC_vals);
//                        }

//                        const int seqlen_remaining = params.height - chunk * kChunkSize - threadIdx.x;

//                        weight_t * dB_cur =
//                                dB + state_idx * params.dB_dstate_stride + chunk * kChunkSize + threadIdx.x;
//                        weight_t * dC_cur =
//                                dC + state_idx * params.dC_dstate_stride + chunk * kChunkSize + threadIdx.x;

                    weight_t * dB_cur = dB + state_idx * params.dB_dstate_stride;
                    weight_t * dC_cur = dC + state_idx * params.dC_dstate_stride;

                    #pragma unroll
                    for (int yy = 0; yy < kThreadSpan; ++yy)
                    {
                        #pragma unroll
                        for (int xx = 0; xx < kThreadSpan; ++xx)
                        {
                            const int gx = thread_x + xx;
                            const int gy = thread_y + yy;

                            if (gx < params.width && gy < params.height)
                            {
                                if constexpr (kIsVariableB)
                                {
                                    dB_cur[gy * params.width + gx] += dB_vals[yy][xx];
                                    // gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals[i]);
                                }

                                if constexpr (kIsVariableC)
                                {
                                    dC_cur[gy * params.width + gx] += dC_vals[yy][xx];
                                    // gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals[i]);
                                }
                            }
                        }
                    }
                }

                __syncthreads();

                if constexpr (!kIsVariableB || !kIsVariableC)
                {
                    // At least one of B and C is const.
                    float2 dA_dBC_val = make_float2(dA_val, dBC_val);
                    dA_dBC_val = typename Kernel::BlockReduce(reduceTempStorage).Sum(dA_dBC_val);
                    dA_val = dA_dBC_val.x;

                    if (threadIdx.x == 0 && threadIdx.y == 0)
                    {
                        // TODO chunk
//                            smem_dbc[state_idx] =
//                                    chunk == params.n_chunks - 1 ?
//                                    dA_dBC_val.y :
//                                    dA_dBC_val.y + smem_dbc[state_idx];

//                        smem_dbc[state_idx] = dA_dBC_val.y;

                        smem_dbc[state_idx] =
                                chunk_y_idx == numChunksAcrossDimY - 1 && chunk_x_idx == numChunksAcrossDimX - 1
                                ? dA_dBC_val.y
                                : dA_dBC_val.y + smem_dbc[state_idx];
                    }
                }
                else
                {
                    dA_val = typename Kernel::BlockReduceFloat(reduceFloatTempStorage).Sum(dA_val);
                }

                if (threadIdx.x == 0 && threadIdx.y == 0)
                {
                    // TODO chunk
//                        smem_da[state_idx] =
//                                chunk == params.n_chunks - 1 ?
//                                dA_val :
//                                dA_val + smem_da[state_idx];

//                    smem_da[state_idx] = dA_val;

                    smem_da[state_idx] =
                                chunk_y_idx == numChunksAcrossDimY - 1 && chunk_x_idx == numChunksAcrossDimX - 1
                                ? dA_val
                                : dA_val + smem_da[state_idx];
                }

            }  // for state_idx

            if constexpr (kDeltaSoftplus)
            {
                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        const int gx = thread_x + xx;
                        const int gy = thread_y + yy;

//                         if (gx < params.width && gy < params.height)
//                         {
//                             delta_vals_load[yy][xx] = delta[gy * params.width + gx];
//                         }
//                         else
//                         {
//                             delta_vals_load[yy][xx] = 1;
//                         }
//
//                         float delta_val = float(delta_vals_load[yy][xx]) + delta_bias;
                        float delta_val = gx < params.width && gy < params.height
                                ? float(delta[gy * params.width + gx]) + delta_bias
                                : 1.0 + delta_bias;
                        float delta_val_neg_exp = expf(-delta_val);

                        ddelta_vals[yy][xx] = delta_val <= 20.0f
                                              ? ddelta_vals[yy][xx] / (1.0f + delta_val_neg_exp)
                                              : ddelta_vals[yy][xx];
                    }
                }
            }

            #pragma unroll
            for (int yy = 0; yy < kThreadSpan; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < kThreadSpan; ++xx)
                {
                    ddelta_bias_val += ddelta_vals[yy][xx];
                }
            }

            // Store du and ddelta to HBM
            #pragma unroll
            for (int yy = 0; yy < kThreadSpan; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < kThreadSpan; ++xx)
                {
                    const int gx = thread_x + xx;
                    const int gy = thread_y + yy;

                    if (gx < params.width && gy < params.height)
                    {
                        du[gy * params.width + gx] = du_vals[yy][xx];
                        ddelta[gy * params.width + gx] = ddelta_vals[yy][xx];
                    }

                    #ifdef NAN_GRAD_CHECK
                    if (isnan(du_vals[yy][xx]))
                    {
                        printf("CUDA Kernel - RuntimeError: "
                               "NaN du_vals detected at "
                               "selective_scan/selective_sacn_bwd_kernel.cuh:1312 "
                               "(before final store to HBM)!"
                               "block (%d %d) thread (%d %d)\n",
                               blockIdx.x, blockIdx.y,
                               threadIdx.x, threadIdx.y);
                        __trap();
                    }
                    if (isnan(ddelta_vals[yy][xx]))
                    {
                        printf("CUDA Kernel - RuntimeError: "
                               "NaN ddelta_vals detected at "
                               "selective_scan/selective_sacn_bwd_kernel.cuh:1323 "
                               "(before final store to HBM)!"
                               "block (%d %d) thread (%d %d)\n",
                               blockIdx.x, blockIdx.y,
                               threadIdx.x, threadIdx.y);
                        __trap();
                    }
                    #endif  // NAN_GRAD_CHECK
                }
            }

        }  // end for chunk_x_idx
    }  // end for chunk_y_idx

    if (params.dD_ptr != nullptr)
    {
        __syncthreads();

        dD_val = typename Kernel::BlockReduceFloat(reduceFloatTempStorage).Sum(dD_val);

        #ifdef NAN_GRAD_CHECK
        if (isnan(dD_val))
        {
            printf("CUDA Kernel - RuntimeError: "
                   "NaN dD_val detected at "
                   "selective_scan/selective_sacn_bwd_kernel.cuh:1348 "
                   "(after block reduction, before final store to HBM)!"
                   "block (%d %d) thread (%d %d)\n",
                   blockIdx.x, blockIdx.y,
                   threadIdx.x, threadIdx.y);
            __trap();
        }
        #endif  // NAN_GRAD_CHECK

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            gpuAtomicAdd(dD, dD_val);
        }
    }

    __syncthreads();

    if (params.ddelta_bias_ptr != nullptr)
    {
        __syncthreads();

        ddelta_bias_val = typename Kernel::BlockReduceFloat(reduceFloatTempStorage).Sum(ddelta_bias_val);

        #ifdef NAN_GRAD_CHECK
        if (isnan(ddelta_bias_val))
        {
            printf("CUDA Kernel - RuntimeError: "
                   "NaN ddelta_bias_val detected at "
                   "selective_scan/selective_sacn_bwd_kernel.cuh:1376 "
                   "(after block reduction, before final store to HBM)!"
                   "block (%d %d) thread (%d %d)\n",
                   blockIdx.x, blockIdx.y,
                   threadIdx.x, threadIdx.y);
            __trap();
        }
        #endif  // NAN_GRAD_CHECK

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            gpuAtomicAdd(ddelta_bias, ddelta_bias_val);
        }
    }

    // ndmamba:
    // reduceFloatTempStorage is __shared__, sync here.
    // Added to also to avoid gpu atomic adds on dB and dC.
    __syncthreads();

    for (int state_idx = threadIdx.y * kBlockDim + threadIdx.x;
         state_idx < params.dstate;
         state_idx += kBlockDim * kBlockDim)
    {
        #ifdef NAN_GRAD_CHECK
        if (isnan(smem_da[state_idx]))
        {
            printf("CUDA Kernel - RuntimeError: "
                   "NaN smem_da[state_idx=%d] detected at "
                   "selective_scan/selective_sacn_bwd_kernel.cuh:1376 "
                   "(after block reduction, before final store to HBM)!"
                   "block (%d %d) thread (%d %d)\n",
                   state_idx,
                   blockIdx.x, blockIdx.y,
                   threadIdx.x, threadIdx.y);
            __trap();
        }
        #endif  // NAN_GRAD_CHECK

        gpuAtomicAdd(&(dA[state_idx * params.dA_dstate_stride]), smem_da[state_idx]);

        weight_t dBC_val;

        if (!kIsVariableB || !kIsVariableC)
        {
            dBC_val = smem_dbc[state_idx];
        }

        if constexpr (!kIsVariableB)
        {
            gpuAtomicAdd(&(dB[state_idx * params.dB_dstate_stride]),
                         !kIsVariableC
                         ? dBC_val * conj(C[state_idx * params.C_dstate_stride])
                         : dBC_val);
        }

        if constexpr (!kIsVariableC)
        {
            gpuAtomicAdd(&(dC[state_idx * params.dC_dstate_stride]),
                         !kIsVariableB
                         ? dBC_val * conj(B[state_idx * params.B_dstate_stride])
                         : dBC_val);
        }
    }
}
#endif  // false


template <int kBlockDim, int kThreadSpan, typename input_t, typename weight_t, typename output_t>
void selective_scan_bwd_launch(SSMParamsBwd & params, cudaStream_t stream)
{
    BOOL_SWITCH(params.width % (kBlockDim * kThreadSpan) == 0 &&
                params.height % (kBlockDim * kThreadSpan) == 0,
                kIsEvenLen,
                [&]
    {
        BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&]
        {
            BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&]
            {
                BOOL_SWITCH(params.delta_softplus, kDeltaSoftplus, [&]
                {
                    BOOL_SWITCH(params.z_ptr != nullptr, kHasZ, [&]
                    {
                        using KernelTraits = SelectiveScanBwdKernelTraits<
                                kBlockDim,
                                kThreadSpan,
                                kIsEvenLen,
                                kIsVariableB,
                                kIsVariableC,
                                kDeltaSoftplus,
                                kHasZ,
                                input_t,
                                weight_t,
                                output_t
                        >;

                        // Running prefixes (states) are stored and loaded from HBM because they're too large.
                        // Also stores dB and dC aggregates in SRAM.
                        const int smemSize = KernelTraits::kSmemSize + params.dstate * sizeof(weight_t) * 2;

                        dim3 grid(params.batch, params.dim);

                        auto kernel = &selective_scan_bwd_kernel<KernelTraits>;

                        if (48 * 1024 < smemSize)
                        {
                            C10_CUDA_CHECK(
                                    cudaFuncSetAttribute(
                                            kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            smemSize
                                    )
                            );
                        }

                        if (64 * 1024 < smemSize)
                        {
                            char buf[1024] = {};
                            sprintf(buf,
                                    "CUDA out of memory. "
                                    "Max shared memory per block: %d Byte(s); "
                                    "requested: %d Byte(s)",
                                    64 * 1024,
                                    smemSize);
                            throw std::runtime_error(buf);
                        }

                        dim3 block(KernelTraits::kBlockDim, KernelTraits::kBlockDim);
                        kernel<<<grid, block, smemSize, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });
}


template <typename input_t, typename weight_t, typename output_t>
void selective_scan_bwd_cuda(SSMParamsBwd & params, cudaStream_t stream)
{
//    // Actual number of threads per block: kBlockDim ** 2.
//    // Tiled. global/kMaxBlockSize <- 16!
//    constexpr int kBlockDim = 8;
//    constexpr int kThreadSpan = 2;
//    static_assert(kBlockDim == 8 && 1 < kThreadSpan && kBlockDim * kThreadSpan <= ndmamba::kMaxDimPerBlock);
//    selective_scan_bwd_launch<kBlockDim, kThreadSpan, input_t, weight_t, output_t>(params, stream);

    if (params.width <= 16 && params.height <= 16)
    {
        // Actual number of threads per block: kBlockDim ** 2.
        // TODO: Must let kBlockDim == 8 and 1 < kThreadSpan. Other settings are bugged.
        constexpr int kBlockDim = 8;
        constexpr int kThreadSpan = 2;
        static_assert(kBlockDim == 8 && 1 < kThreadSpan && kBlockDim * kThreadSpan <= ndmamba::kMaxDimPerBlock);
        selective_scan_bwd_launch<kBlockDim, kThreadSpan, input_t, weight_t, output_t>(params, stream);
    }
    else
    {
        // Actual number of threads per block: kBlockDim ** 2.
        // TODO: Must let kBlockDim == 8 and 1 < kThreadSpan. Other settings are bugged.
        constexpr int kBlockDim = 8;
        constexpr int kThreadSpan = 4;
        static_assert(kBlockDim == 8 && 1 < kThreadSpan && kBlockDim * kThreadSpan <= ndmamba::kMaxDimPerBlock);
        selective_scan_bwd_launch<kBlockDim, kThreadSpan, input_t, weight_t, output_t>(params, stream);
    }
}