/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "selective_scan/global.cuh"
#include "selective_scan/selective_scan.cuh"
#include "selective_scan/selective_scan_common.cuh"
#include "selective_scan/static_switch.cuh"

#include "scan/block_scan.cuh"


template <int kBlockDim_, int kThreadSpan_, bool kIsEvenLen_,
        bool kIsVariableB_, bool kIsVariableC_,
        bool kHasZ_, typename input_t_, typename weight_t_, typename output_t_>
struct SelectiveScanFwdKernelTraits
{
    static constexpr int kThreadSpan = kThreadSpan_;
    //    static constexpr int kNItems = kThreadSpan * kThreadSpan;
    //    static_assert(kNItems % 4 == 0);

    using input_t = input_t_;
    using weight_t = weight_t_;
    using output_t = output_t_;

    static constexpr int kBlockDim = kBlockDim_;
    //    static constexpr int kNThreads = kBlockDim_ * kBlockDim_;

    // Oringinal mamba.
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    // static constexpr int kMinBlocks = (kBlockDim * kBlockDim) < 128 ? 5 : 3;

    // ndmamba:
    static constexpr int kMinBlocks = (kBlockDim * kBlockDim) < 128 ? 5 : 2;

    //    static constexpr int kNBytes = sizeof(input_t);
    //    static_assert(kNBytes == 2 || kNBytes == 4);

    //    // kNElts: for fp32:          4
    //    //         for fp16 & others: 4 or 8 ( 8 when seqlen is large)
    //    static constexpr int kNElts = kNBytes == 4 ? 4 : CUB_MIN(8, kNItems);
    //    static_assert(kNItems % kNElts == 0);

    //    // kNLoads: for fp32:   kNItems / 4
    //    //         for others: kNItems / 4 or kNItems / 8 ( kNItems / 8 when seqlen is large)
    //    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsComplex = std::is_same_v<weight_t, complex_t>;
    //    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kHasZ = kHasZ_;

    //    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;

    //    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = std::conditional_t<!kIsComplex, float2, float4>;

    //    using BlockLoadT = cub::BlockLoad<
    //            input_t,
    //            kNThreads,
    //            kNItems,
    //            cub::BLOCK_LOAD_WARP_TRANSPOSE
    //    >;
    //    using BlockLoadVecT = cub::BlockLoad<
    //            vec_t,
    //            kNThreads,
    //            kNLoads,
    //            !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT
    //    >;
    //
    //    using BlockLoadWeightT = cub::BlockLoad<
    //            input_t,
    //            kNThreads,
    //            !kIsComplex ? kNItems : kNItems * 2,
    //            cub::BLOCK_LOAD_WARP_TRANSPOSE
    //    >;
    //    using BlockLoadWeightVecT = cub::BlockLoad<
    //            vec_t,
    //            kNThreads,
    //            !kIsComplex ? kNLoads : kNLoads * 2,
    //            !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT
    //    >;
    //
    //    using BlockStoreT = cub::BlockStore<
    //            input_t,
    //            kNThreads,
    //            kNItems,
    //            cub::BLOCK_STORE_WARP_TRANSPOSE
    //    >;
    //    using BlockStoreVecT = cub::BlockStore<
    //            vec_t,
    //            kNThreads,
    //            kNLoads,
    //            !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT
    //    >;

    //    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    //
    //    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
    //                                                   sizeof(typename BlockLoadVecT::TempStorage),
    //                                                   (int(kIsVariableB) + int(kIsVariableC)) *
    //                                                   sizeof(typename BlockLoadWeightT::TempStorage),
    //                                                   (int(kIsVariableB) + int(kIsVariableC)) *
    //                                                   sizeof(typename BlockLoadWeightVecT::TempStorage),
    //                                                   sizeof(typename BlockStoreT::TempStorage),
    //                                                   sizeof(typename BlockStoreVecT::TempStorage)});
    //
    //    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);

    static constexpr int kSegLen = 8;
    using BlockScan = ndmamba::SegBlockScan<scan_t, kSegLen, kBlockDim, kBlockDim>;
    static constexpr int kSmemSize = sizeof(typename BlockScan::TempStorage);
};


template <typename Kernel>
__global__ __launch_bounds__(Kernel::kBlockDim * Kernel::kBlockDim, Kernel::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params)
{
    constexpr bool kIsComplex = Kernel::kIsComplex;
    constexpr bool kIsVariableB = Kernel::kIsVariableB;
    constexpr bool kIsVariableC = Kernel::kIsVariableC;
    constexpr bool kHasZ = Kernel::kHasZ;

    constexpr int kThreadSpan = Kernel::kThreadSpan;

    using input_t = typename Kernel::input_t;
    using weight_t = typename Kernel::weight_t;
    using scan_t = typename Kernel::scan_t;
    using output_t = typename Kernel::output_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto & scanTempStorage = *reinterpret_cast<typename Kernel::BlockScan::TempStorage *>(smem_);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;

    // ndmamba: Input tensor B is of shape (b 1 dstate l),
    // so the "dim" in dim_ngroups_ratio is 1, NOT embed_dim.
    // group_id * B_group_stride exactly offsets the N dimension.
    const int group_id = dim_id / (params.dim_ngroups_ratio);

    input_t * u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
                  + dim_id * params.u_d_stride;
    input_t * delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
                      + dim_id * params.delta_d_stride;

    weight_t * A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;

    weight_t * B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * params.B_d_stride;
    input_t * Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride +
                     group_id * params.B_group_stride;
    weight_t * C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * params.C_d_stride;
    input_t * Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride +
                     group_id * params.C_group_stride;

    // Size of a square chunk (when there are multiple chunks).
    // Invalid when there's only one chunk.
    constexpr int kChunkDim = ndmamba::kMaxDimPerBlock;

    const int numChunksAcrossDimX = (params.width + kChunkDim - 1) / kChunkDim;
    const int numChunksAcrossDimY = (params.height + kChunkDim - 1) / kChunkDim;

    scan_t * x_ptr = reinterpret_cast<scan_t *>(params.x_ptr) +
            (batch_id * params.dim + dim_id) *
            (numChunksAcrossDimY * numChunksAcrossDimX * 2 * kChunkDim * params.dstate);

    // Used for memory boundary check debugging.
    // batch_size == gridDim.x
    #ifdef BOUNDARY_CHECK
    const int x_len_scan_t = gridDim.x * params.dim * numChunksAcrossDimY * numChunksAcrossDimX * params.dstate * 2 * kChunkDim;
    auto x_boundary_scan_t = reinterpret_cast<scan_t *>(params.x_ptr) + x_len_scan_t;
    #endif  // BOUNDARY_CHECK

    //                x = torch::empty(
    //                        {
    //                                batch_size,
    //                                dim,
    //                                numChunksAcrossDimY,
    //                                numChunksAcrossDimX,
    //                                dstate,
    //                                2,  // horizontal, vertical
    //                                mamband::kMaxDimPerBlock,
    //                                2   // A, Bx
    //                        },
    //                        u.options().dtype(weight_type)  // real, complex
    //                );

    auto x = [
            x_ptr,
            &params,
            kChunkDim,
            kThreadSpan,
            numChunksAcrossDimX,
            numChunksAcrossDimY
    ](
            int chunk_y_idx,
            int chunk_x_idx,
            int state_idx,
            ndmamba::ScanDir scanDir) -> scan_t *
    {
        int offset = (chunk_y_idx * numChunksAcrossDimX + chunk_x_idx) * (params.dstate * kChunkDim * 2) +
                     state_idx * 2 * kChunkDim +
                     scanDir * kChunkDim;

        if (scanDir == ndmamba::kHorizontal)
        {
            // threadIdx.x == 0
            offset += threadIdx.y * kThreadSpan;
        }
        else if (scanDir == ndmamba::kVertical)
        {
            // threadIdx.y == 0
            offset += threadIdx.x * kThreadSpan;
        }
        else
        {
            // TODO scanDir
        }

        return x_ptr + offset;
    };

    float D_val = params.D_ptr ? reinterpret_cast<float *>(params.D_ptr)[dim_id] : 0;
    float delta_bias = params.delta_bias_ptr ? reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id] : 0;

    for (int chunk_y_idx = 0; chunk_y_idx < numChunksAcrossDimY; ++chunk_y_idx)
    {
        for (int chunk_x_idx = 0; chunk_x_idx < numChunksAcrossDimX; ++chunk_x_idx)
        {
            input_t u_vals[kThreadSpan][kThreadSpan];
            input_t delta_vals_load[kThreadSpan][kThreadSpan];

            const int thread_x = kChunkDim * chunk_x_idx + threadIdx.x * kThreadSpan;
            const int thread_y = kChunkDim * chunk_y_idx + threadIdx.y * kThreadSpan;

            // Load input
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
                        delta_vals_load[yy][xx] = delta[gy * params.width + gx];
                        // u[gy * params.width + gx] = threadIdx.y * 100 + threadIdx.x;
                    }
                    else
                    {
                        u_vals[yy][xx] = 0;
                        delta_vals_load[yy][xx] = 1;
                    }
                }
            }

            // delta_vals = softplus(delta + delta_bias)
            // delta_u = delta_vals * u, python: delta = delta(x)
            // out_vals = D * u, python: y = y + D * x
            float delta_vals[kThreadSpan][kThreadSpan];
            float delta_u_vals[kThreadSpan][kThreadSpan];
            float out_vals[kThreadSpan][kThreadSpan];

            #pragma unroll
            for (int yy = 0; yy < kThreadSpan; ++yy)
            {
                #pragma unroll
                for (int xx = 0; xx < kThreadSpan; ++xx)
                {
                    auto u_val = static_cast<float>(u_vals[yy][xx]);
                    delta_vals[yy][xx] = static_cast<float>(delta_vals_load[yy][xx]) + delta_bias;

                    if (params.delta_softplus)
                    {
                        delta_vals[yy][xx] =
                                delta_vals[yy][xx] <= 20.0f ?
                                log1pf(expf(delta_vals[yy][xx])) :
                                delta_vals[yy][xx];
                    }

                    delta_u_vals[yy][xx] = delta_vals[yy][xx] * u_val;
                    out_vals[yy][xx] = D_val * u_val;  // residual

//                    const int gx = thread_x + xx;
//                    const int gy = thread_y + yy;
//
//                    if (gx < params.width && gy < params.height)
//                    {
//                        u[gy * params.width + gx] = delta_u_vals[yy][xx];
//                    }
                }
            }

            // Sequential processing over number of heads, default 16.
            for (int state_idx = 0; state_idx < params.dstate; ++state_idx)
            {
                __syncthreads();

                weight_t A_val = A[state_idx * params.A_dstate_stride]; // read A, A: D, N
                constexpr float kLog2e = M_LOG2E;

                // take log for A
                if constexpr (!kIsComplex)
                {
                    A_val *= kLog2e;
                }
                else
                {
                    A_val.real_ *= kLog2e;
                }

                // This variable holds B * C if both B and C are constant across seqlen.
                // If only B varies across seqlen, this holds C.
                // If only C varies across seqlen, this holds B.
                // If both B and C vary, this is unused.
                weight_t BC_val;
                weight_t B_vals[kThreadSpan][kThreadSpan];
                weight_t C_vals[kThreadSpan][kThreadSpan];

                // BC_vals stores C (constant [1]), B_vals stores B ([kThreadSpan][kThreadSpan])
                if constexpr (kIsVariableB)
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

                    if constexpr (!kIsVariableC)  // C is constant
                    {
                        BC_val = C[state_idx * params.C_dstate_stride];
                    }
                }

                if constexpr (kIsVariableC)
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
                                C_vals[yy][xx] = Cvar[state_idx * params.B_dstate_stride + gy * params.width + gx];
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

                    if constexpr (!kIsVariableB)  // B is constant
                    {
                        BC_val = B[state_idx * params.B_dstate_stride];
                    }
                }

                if constexpr (!kIsVariableB && !kIsVariableC)
                {
                    BC_val = B[state_idx * params.B_dstate_stride] * C[state_idx * params.C_dstate_stride];
                }

                // Scan
                __syncthreads();
                scan_t thread_data[kThreadSpan][kThreadSpan];

                // initilize thread_data.x = deltaA.y = B*x
                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        const int gx = thread_x + xx;
                        const int gy = thread_y + yy;

                        if constexpr (!kIsComplex)
                        {
                            if (gx < params.width && gy < params.height)
                            {
                                thread_data[yy][xx] = make_float2(
                                    exp2f(delta_vals[yy][xx] * A_val),
                                    !kIsVariableB ?
                                    delta_u_vals[yy][xx] :
                                    B_vals[yy][xx] * delta_u_vals[yy][xx]
                                );
                            }
                            else
                            {
                                // So that the last state is correct
                                thread_data[yy][xx] = make_float2(1.0f, 0.0f);
                            }
                        }
                        else
                        {
                            if (gx < params.width && gy < params.height)
                            {
                                // Pytorch's implementation of complex exp (which calls thrust) is very slow
                                complex_t delta_a_exp = cexp2f(delta_vals[yy][xx] * A_val);
                                weight_t B_delta_u_val = !kIsVariableB ?
                                                         delta_u_vals[yy][xx] :
                                                         B_vals[yy][xx] * delta_u_vals[yy][xx];
                                thread_data[yy][xx] = make_float4(delta_a_exp.real_,
                                                                  delta_a_exp.imag_,
                                                                  B_delta_u_val.real_,
                                                                  B_delta_u_val.imag_);
                            }
                            else
                            {
                                // So that the last state is correct
                                thread_data[yy][xx] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
                            }
                        }
                    }
                }

                // +-------------- X
                // |
                // |
                // |
                // |
                // Y

                // Initialize running total to process multiple chunk,
                // succeeding chunks need the results of the 1st chunk
                scan_t running_prefixes_for_horizontal_scan[kThreadSpan];
                scan_t running_prefixes_for_vertical_scan[kThreadSpan];

                scan_t * hor_scan_prefixes_to_read = x(chunk_y_idx, chunk_x_idx - 1, state_idx, ndmamba::kHorizontal);
                scan_t * ver_scan_prefixes_to_read = x(chunk_y_idx - 1, chunk_x_idx, state_idx, ndmamba::kVertical);

                #pragma unroll
                for (int yx = 0; yx < kThreadSpan; ++yx)
                {
                    if constexpr (!kIsComplex)
                    {
                        running_prefixes_for_horizontal_scan[yx] =
                                0 < chunk_x_idx && threadIdx.x == 0 ?
                                hor_scan_prefixes_to_read[yx] :
                                make_float2(1.0f, 0.0f);

                        running_prefixes_for_vertical_scan[yx] =
                                0 < chunk_y_idx && threadIdx.y == 0 ?
                                ver_scan_prefixes_to_read[yx] :
                                make_float2(1.0f, 0.0f);
                    }
                    else
                    {
                        running_prefixes_for_horizontal_scan[yx] =
                                0 < chunk_x_idx && threadIdx.x == 0 ?
                                hor_scan_prefixes_to_read[yx] :
                                make_float4(1.0f, 0.0f, 0.0f, 0.0f);

                        running_prefixes_for_vertical_scan[yx] =
                                0 < chunk_y_idx && threadIdx.y == 0 ?
                                ver_scan_prefixes_to_read[yx] :
                                make_float4(1.0f, 0.0f, 0.0f, 0.0f);
                    }
                }

                SSMScanPrefixCallbackOp<weight_t> prefix_ops_for_horizontal_scan[kThreadSpan];
                SSMScanPrefixCallbackOp<weight_t> prefix_ops_for_vertical_scan[kThreadSpan];

                #pragma unroll
                for (int yx = 0; yx < kThreadSpan; ++yx)
                {
                    prefix_ops_for_horizontal_scan[yx].running_prefix = running_prefixes_for_horizontal_scan[yx];
                    prefix_ops_for_vertical_scan[yx].running_prefix = running_prefixes_for_vertical_scan[yx];
                }

                typename Kernel::BlockScan scan(scanTempStorage);

                __syncthreads();

                scan.InclusiveScan(thread_data,
                                   thread_data,
                                   SSMScanOp<weight_t>(),
                                   prefix_ops_for_horizontal_scan,
                                   ndmamba::kHorizontal);

                __syncthreads();

                // Re-initialize A
                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        const int gx = thread_x + xx;
                        const int gy = thread_y + yy;

                        if constexpr (!kIsComplex)
                        {
                            if (gx < params.width && gy < params.height)
                            {
                                thread_data[yy][xx].x = exp2f(delta_vals[yy][xx] * A_val);
                            }
                            else
                            {
                                // So that the last state is correct
                                thread_data[yy][xx].x = 1.0f;
                            }
                        }
                        else
                        {
                            if (gx < params.width && gy < params.height)
                            {
                                // Pytorch's implementation of complex exp (which calls thrust) is very slow
                                complex_t delta_a_exp = cexp2f(delta_vals[yy][xx] * A_val);
                                thread_data[yy][xx].x = delta_a_exp.real_;
                                thread_data[yy][xx].y = delta_a_exp.imag_;
                            }
                            else
                            {
                                // So that the last state is correct
                                thread_data[yy][xx].x = 1.0f;
                                thread_data[yy][xx].y = 0.0f;
                            }
                        }
                    }
                }

                scan.InclusiveScan(thread_data,
                                   thread_data,
                                   SSMScanOp<weight_t>(),
                                   prefix_ops_for_vertical_scan,
                                   ndmamba::kVertical);

                __syncthreads();

                // There's a syncthreads in the scan op, so we don't need to sync here.
                // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.

//                x = torch::empty(
//                        {
//                                batch_size,
//                                dim,
//                                numChunksDimY,
//                                numChunksDimX,
//                                dstate,
//                                mamband::kMaxDimPerBlock,
//                                2,  // horizontal, vertical
//                                2   // A, Bx
//                        },
//                        u.options().dtype(weight_type)  // real, complex
//                );

                scan_t * hor_scan_prefixes_to_write = x(chunk_y_idx, chunk_x_idx, state_idx, ndmamba::kHorizontal);
                scan_t * ver_scan_prefixes_to_write = x(chunk_y_idx, chunk_x_idx, state_idx, ndmamba::kVertical);

                #pragma unroll
                for (int yx = 0; yx < kThreadSpan; ++yx)
                {
                    if (threadIdx.x == 0)
                    {
                        // Horizontal scan.
                        #ifdef BOUNDARY_CHECK
                        if (x_boundary_scan_t <= hor_scan_prefixes_to_write + yx)
                        {
                            printf("CUDA out-of-bound HBM store @ %p (boundary @ %p) "
                                   "by block (%d %d) thread (%d %d) chunk (%d %d)\n",
                                   hor_scan_prefixes_to_write + yx,
                                   x_boundary_scan_t,
                                   blockIdx.x, blockIdx.y,
                                   threadIdx.x, threadIdx.y,
                                   chunk_x_idx, chunk_y_idx);
                            __trap();
                        }
                        #endif  // BOUNDARY_CHECK

                        hor_scan_prefixes_to_write[yx] = prefix_ops_for_horizontal_scan[yx].running_prefix;
                    }

                    if (threadIdx.y == 0)
                    {
                        // Vertical scan.
                        #ifdef BOUNDARY_CHECK
                        if (x_boundary_scan_t <= ver_scan_prefixes_to_write + yx)
                        {
                            printf("CUDA out-of-bound HBM store @ %p (boundary @ %p) "
                                   "by block (%d %d) thread (%d %d) chunk (%d %d)\n",
                                   ver_scan_prefixes_to_write + yx,
                                   x_boundary_scan_t,
                                   blockIdx.x, blockIdx.y,
                                   threadIdx.x, threadIdx.y,
                                   chunk_x_idx, chunk_y_idx);
                            __trap();
                        }
                        #endif  // BOUNDARY_CHECK

                        ver_scan_prefixes_to_write[yx] = prefix_ops_for_vertical_scan[yx].running_prefix;
                    }
                }

                // multiply by C
                #pragma unroll
                for (int yy = 0; yy < kThreadSpan; ++yy)
                {
                    #pragma unroll
                    for (int xx = 0; xx < kThreadSpan; ++xx)
                    {
                        const weight_t C_val =
                                !kIsVariableC ?
                                BC_val :
                                (!kIsVariableB ? BC_val * C_vals[yy][xx] : C_vals[yy][xx]);

                        if constexpr (!kIsComplex)
                        {
                            out_vals[yy][xx] += thread_data[yy][xx].y * C_val;
                        }
                        else
                        {
                            out_vals[yy][xx] += (complex_t(thread_data[yy][xx].z, thread_data[yy][xx].w) * C_val).real_ * 2;
                        }
                    }
                }
            }  // for state_idx

            // Write-back.
            output_t * out = reinterpret_cast<output_t *>(params.out_ptr) +
                            batch_id * params.out_batch_stride +
                            dim_id * params.out_d_stride;

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
                        out[gy * params.width + gx] = static_cast<output_t>(out_vals[yy][xx]);
                    }
                }
            }

            if constexpr (kHasZ)
            {
                input_t * z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                              + dim_id * params.z_d_stride;
                output_t * out_z = reinterpret_cast<output_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                                  + dim_id * params.out_z_d_stride;

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
                            float z_val = z[gy * params.width + gx];
                            out_vals[yy][xx] *= z_val / (1 + expf(-z_val));
                            out_z[gy * params.width + gx] = static_cast<output_t>(out_vals[yy][xx]);
                        }
                    }
                }
            }
        }
    }
}


template <int kBlockDim, int kThreadSpan, typename input_t, typename weight_t, typename output_t>
void selective_scan_fwd_launch(SSMParamsBase & params, cudaStream_t stream)
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
                            BOOL_SWITCH(params.z_ptr != nullptr, kHasZ, [&]
                            {
                                using KernelTraits = SelectiveScanFwdKernelTraits<
                                        kBlockDim,
                                        kThreadSpan,
                                        kIsEvenLen,
                                        kIsVariableB,
                                        kIsVariableC,
                                        kHasZ,
                                        input_t,
                                        weight_t,
                                        output_t
                                >;

                                // Running prefixes (states) are stored and loaded from HBM because they're too large
                                const int smemSize = KernelTraits::kSmemSize;

                                dim3 grid(params.batch, params.dim);

                                // Had to change this substantially since potentially the hip
                                // interface for setting kernel launch attributes is slightly different from
                                // cuda's. In particualar, it seems to expect a plain const void * pointer.

                                auto kernel = &selective_scan_fwd_kernel<KernelTraits>;

                                if (smemSize >= 48 * 1024)
                                {
                                    C10_CUDA_CHECK(
                                            cudaFuncSetAttribute(
                                                    kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    smemSize
                                            )
                                    );
                                }

                                // kNThreads is an int, not a dim3, so threadIdx.y/z == 0
                                dim3 block(KernelTraits::kBlockDim, KernelTraits::kBlockDim);
                                kernel<<<grid, block, smemSize, stream>>>(params);
                                C10_CUDA_KERNEL_LAUNCH_CHECK();
//                                C10_CUDA_CHECK(cudaDeviceSynchronize());
                            });
                        });
                    });
                });
}


template <typename input_t, typename weight_t, typename output_t>
void selective_scan_fwd_cuda(SSMParamsBase & params, cudaStream_t stream)
{
//    // Actual number of threads per block: kBlockDim ** 2.
//    // Tiled. global/kMaxBlockSize <- 16!
//    constexpr int kBlockDim = 8;
//    constexpr int kThreadSpan = 2;
//    static_assert(kBlockDim == 8 && 1 < kThreadSpan && kBlockDim * kThreadSpan <= ndmamba::kMaxDimPerBlock);
//    selective_scan_fwd_launch<kBlockDim, kThreadSpan, input_t, weight_t, output_t>(params, stream);

    if (params.width <= 16 && params.height <= 16)
    {
        // Actual number of threads per block: kBlockDim ** 2.
        // TODO: Must let kBlockDim == 8 and 1 < kThreadSpan. Other settings are bugged.
        constexpr int kBlockDim = 8;
        constexpr int kThreadSpan = 2;
        static_assert(kBlockDim == 8 && 1 < kThreadSpan && kBlockDim * kThreadSpan <= ndmamba::kMaxDimPerBlock);
        selective_scan_fwd_launch<kBlockDim, kThreadSpan, input_t, weight_t, output_t>(params, stream);
    }
    else
    {
        // Actual number of threads per block: kBlockDim ** 2.
        // TODO: Must let kBlockDim == 8 and 1 < kThreadSpan. Other settings are bugged.
        constexpr int kBlockDim = 8;
        constexpr int kThreadSpan = 4;
        static_assert(kBlockDim == 8 && 1 < kThreadSpan && kBlockDim * kThreadSpan <= ndmamba::kMaxDimPerBlock);
        selective_scan_fwd_launch<kBlockDim, kThreadSpan, input_t, weight_t, output_t>(params, stream);
    }
}
