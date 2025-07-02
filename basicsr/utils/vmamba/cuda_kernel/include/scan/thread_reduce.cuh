#pragma once

#include "scan/commons.h"


namespace ndmamba
{

/**
 * @brief Serial reduction with the specified operator
 *
 * @tparam LENGTH
 *   <b>[inferred]</b> LengthT of @p input array
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be reduced.
 *
 * @tparam ReductionOp
 *   <b>[inferred]</b> Binary reduction operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[in] reduction_op
 *   Binary reduction operator
 */
template <int LENGTH, typename T, typename ReductionOp>
__device__ __forceinline__ void ThreadReduce(T (& input)[LENGTH][LENGTH],
                                             T (& output)[LENGTH],
                                             ReductionOp reduction_op,
                                             ScanDir scan_dir)
{
    if (scan_dir == kHorizontal)
    {
        #pragma unroll
        for (int r = 0; r < LENGTH; ++r)
        {
            output[r] = input[r][0];
        }

        #pragma unroll
        for (int r = 0; r < LENGTH; ++r)
        {
            #pragma unroll
            for (int c = 1; c < LENGTH; ++c)
            {
                output[r] = reduction_op(output[r], input[r][c]);
            }
        }
    }
    else if (scan_dir == kVertical)
    {
        #pragma unroll
        for (int c = 0; c < LENGTH; ++c)
        {
            output[c] = input[0][c];
        }

        #pragma unroll
        for (int c = 0; c < LENGTH; ++c)
        {
            #pragma unroll
            for (int r = 1; r < LENGTH; ++r)
            {
                output[c] = reduction_op(output[c], input[r][c]);
            }
        }
    }
    else if (scan_dir == kHorizontalReversed)
    {
        #pragma unroll
        for (int r = 0; r < LENGTH; ++r)
        {
            output[r] = input[r][LENGTH - 1];
        }

        #pragma unroll
        for (int r = 0; r < LENGTH; ++r)
        {
            #pragma unroll
            for (int c = LENGTH - 2; 0 <= c; --c)
            {
                output[r] = reduction_op(output[r], input[r][c]);
            }
        }
    }
    else if (scan_dir == kVerticalReversed)
    {
        #pragma unroll
        for (int c = 0; c < LENGTH; ++c)
        {
            output[c] = input[LENGTH - 1][c];
        }

        #pragma unroll
        for (int c = 0; c < LENGTH; ++c)
        {
            #pragma unroll
            for (int r = LENGTH - 2; 0 <= r; --r)
            {
                output[c] = reduction_op(output[c], input[r][c]);
            }
        }
    }
    else
    {
        __trap();
    }
}

}  // namespace ndmamba

