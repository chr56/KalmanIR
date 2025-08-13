/**
 * @file
 * Thread utilities for sequential prefix scan over statically-sized array types
 */

#pragma once


namespace ndmamba
{

/**
 * @brief Perform a sequential inclusive prefix scan over the
 *        statically-sized @p input array, seeded with the specified @p prefix.
 *        The aggregate is returned.
 *
 * @tparam LENGTH
 *   <b>[inferred]</b> LengthT of @p input and @p output arrays
 *
 * @tparam T
 *   <b>[inferred]</b> The data type to be scanned.
 *
 * @tparam ScanOp
 *   <b>[inferred]</b> Binary scan operator type having member
 *   <tt>T operator()(const T &a, const T &b)</tt>
 *
 * @param[in] input
 *   Input array
 *
 * @param[out] output
 *   Output array (may be aliased to @p input)
 *
 * @param[in] scan_op
 *   Binary scan operator
 *
 * @param[in] prefix
 *   Prefix to seed scan with
 *
 * @param[in] apply_prefix
 *   Whether or not the calling thread should apply its prefix.
 *   (Handy for preventing thread-0 from applying a prefix.)
 */
template <int LENGTH, typename T, typename ScanOp>
__device__ __forceinline__ void ThreadScanInclusive(T (& input)[LENGTH][LENGTH],
                                                    T (& output)[LENGTH][LENGTH],
                                                    ScanOp scan_op,
                                                    T (& prefixes)[LENGTH],
                                                    bool apply_prefix,
                                                    ScanDir scanDir)
{
    if (scanDir == kHorizontal)
    {
        #pragma unroll
        for (int row = 0; row < LENGTH; ++row)
        {
            output[row][0] = apply_prefix ? scan_op(prefixes[row], input[row][0]) : input[row][0];
        }

        #pragma unroll
        for (int row = 0; row < LENGTH; ++row)
        {
            #pragma unroll
            for (int col = 1; col < LENGTH; ++col)
            {
                output[row][col] = scan_op(output[row][col - 1], input[row][col]);
            }
        }
    }
    else if (scanDir == kVertical)
    {
        #pragma unroll
        for (int col = 0; col < LENGTH; ++col)
        {
            output[0][col] = apply_prefix ? scan_op(prefixes[col], input[0][col]) : input[0][col];
        }

        #pragma unroll
        for (int col = 0; col < LENGTH; ++col)
        {
            #pragma unroll
            for (int row = 1; row < LENGTH; ++row)
            {
                output[row][col] = scan_op(output[row - 1][col], input[row][col]);
            }
        }
    }
    else if (scanDir == kHorizontalReversed)
    {
        #pragma unroll
        for (int row = 0; row < LENGTH; ++row)
        {
            output[row][LENGTH - 1] =
                    apply_prefix ?
                    scan_op(prefixes[row], input[row][LENGTH - 1]) :
                    input[row][LENGTH - 1];
        }

        #pragma unroll
        for (int row = 0; row < LENGTH; ++row)
        {
            #pragma unroll
            for (int col = LENGTH - 2; 0 <= col; --col)
            {
                output[row][col] = scan_op(output[row][col + 1], input[row][col]);
            }
        }
    }
    else if (scanDir == kVerticalReversed)
    {
        #pragma unroll
        for (int col = 0; col < LENGTH; ++col)
        {
            output[LENGTH - 1][col] =
                    apply_prefix ?
                    scan_op(prefixes[col], input[LENGTH - 1][col]) :
                    input[LENGTH - 1][col];
        }

        #pragma unroll
        for (int col = 0; col < LENGTH; ++col)
        {
            #pragma unroll
            for (int row = LENGTH - 2; 0 <= row; --row)
            {
                output[row][col] = scan_op(output[row + 1][col], input[row][col]);
            }
        }
    }
    else
    {
        __trap();
    }
}

}  // namespace ndmamba

