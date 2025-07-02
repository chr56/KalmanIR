#pragma once

#include <cub/config.cuh>
#include <cub/util_ptx.cuh>

#include "warp_scan.cuh"


namespace ndmamba
{

/**
 * \brief BlockScanWarpScans provides warpscan-based variants of parallel prefix scan across a CUDA thread block.
 */
template <
        typename T,
        int kSegLen,
        int BLOCK_DIM_X,    ///< The thread block length in threads along the X dimension
        int BLOCK_DIM_Y,    ///< The thread block length in threads along the Y dimension
        int BLOCK_DIM_Z,    ///< The thread block length in threads along the Z dimension
        int PTX_ARCH       ///< The PTX compute capability for which to to specialize this collective
>
struct SegBlockScanWarpScans
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        /// Number of warp threads
        WARP_THREADS = CUB_WARP_THREADS(PTX_ARCH),

        /// The thread block size in threads
        BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,

        /// Number of active warps
        WARPS = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,

        /// Warp shape
        kWarpDimX = kSegLen,
        kWarpDimY = WARP_THREADS / kSegLen,

        /// Number of warps (col x kSegLen) across block's x dim
        kWarpsAcrossBlockDimX = BLOCK_DIM_X / kWarpDimX,

        /// Number of warps (col x kSegLen) across block's y dim
        kWarpsAcrossBlockDimY = BLOCK_DIM_Y / kWarpDimY,
    };

    ///  WarpScan utility type
    typedef SegWarpScan<T, kSegLen, WARP_THREADS, PTX_ARCH> WarpScanT;

    #if false
    ///  WarpScan utility type
    typedef SegWarpScan<T, kSegLen, WARPS, PTX_ARCH> WarpAggregateScan;
    #endif  // false

    /// Shared memory storage layout type

    struct __align__(32) InternalTempStorage
    {
        T warp_aggregates[WARPS][CUB_MAX(WARP_THREADS / kSegLen, kSegLen)];
        typename WarpScanT::TempStorage warp_scan[WARPS];           ///< Buffer for warp-synchronous scans
    };


    /// Alias wrapper allowing storage to be unioned
    struct TempStorage : cub::Uninitialized<InternalTempStorage>
    {
    };


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------

    // Thread fields
    InternalTempStorage & temp_storage;

    unsigned int linear_tid;

    unsigned int warp_id;

    unsigned int lane_id;


    //---------------------------------------------------------------------
    // Constructors
    //---------------------------------------------------------------------

    /// Constructor
    __device__ __forceinline__ SegBlockScanWarpScans(
            TempStorage & temp_storage)
            :
            temp_storage(temp_storage.Alias()),
            linear_tid(cub::RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z)),
            warp_id((WARPS == 1) ? 0 : linear_tid / WARP_THREADS),
            lane_id(cub::LaneId())
    {
    }


    //---------------------------------------------------------------------
    // Utility methods
    //---------------------------------------------------------------------

    #if false
    template <typename ScanOp, int WARP>
    __device__ __forceinline__ void ApplyWarpAggregates(
            T & warp_prefix,           ///< [out] The calling thread's partial reduction
            ScanOp scan_op,            ///< [in] Binary scan operator
            T & block_aggregate,   ///< [out] Threadblock-wide aggregate reduction of input items
            cub::Int2Type<WARP>  /*addend_warp*/)
    {
        if (warp_id == WARP)
        {
            warp_prefix = block_aggregate;
        }

        T addend = temp_storage.warp_aggregates[WARP];
        block_aggregate = scan_op(block_aggregate, addend);

        ApplyWarpAggregates(warp_prefix, scan_op, block_aggregate, cub::Int2Type<WARP + 1>());
    }

    template <typename ScanOp>
    __device__ __forceinline__ void ApplyWarpAggregates(
            T &/*warp_prefix*/,       ///< [out] The calling thread's partial reduction
            ScanOp          /*scan_op*/,            ///< [in] Binary scan operator
            T &/*block_aggregate*/,   ///< [out] Threadblock-wide aggregate reduction of input items
            cub::Int2Type<WARPS> /*addend_warp*/)
    {
    }
    #endif  // false

    /// Use the warp-wide aggregates to compute the calling warp's prefix.
    /// Also returns block-wide aggregate in all threads.
    /// NEW for ndmamba: Each warp has size (kWarpDimX, kWarpDimY).
    /// Each row or column has its own block_aggregate (computed sequentially)
    /// All threads of that row or column do the same computation,
    /// and populate its own warp prefix when looping to its own location.)
    /// The warp_prefix for the first row or the first column is ZERO.
    /// block_aggregate is INCLUSIVE w.r.t. warp_aggregate.
    template <typename ScanOp>
    __device__ __forceinline__ T ComputeWarpPrefix(
            ScanOp scan_op,            ///< [in] Binary scan operator
            T warp_aggregate,     ///< [in] <b>[<em>lane</em><sub>WARP_THREADS - 1</sub> only]</b> Warp-wide aggregate reduction of input items
            T & block_aggregate,   ///< [out] Threadblock-wide aggregate reduction of input items
            ScanDir scanDir)      ///< [in] Scan direction
    {
        // IMPORTANT! warp_prefix set to zero for first-row or first-column threads in the whole block!
        T warp_prefix = {};

        int xInWarp = lane_id % kWarpDimX;
        int yInWarp = lane_id / kWarpDimX;

        // Added to pass compute sanitizer racecheck.
        cub::CTA_SYNC();

        // Leading/trailing thread in each row or column write its inclusive_output
        // to __shared__ temp_storage warp aggregate.
        if (scanDir == kHorizontal)
        {
            if ((lane_id + 1) % kWarpDimX == 0)
            {
                temp_storage.warp_aggregates[warp_id][yInWarp] = warp_aggregate;
            }
        }
        else if (scanDir == kVertical)
        {
            if ((WARP_THREADS <= lane_id + kWarpDimX))
            {
                temp_storage.warp_aggregates[warp_id][xInWarp] = warp_aggregate;
            }
        }
        else if (scanDir == kHorizontalReversed)
        {
            if (lane_id % kWarpDimX == 0)
            {
                temp_storage.warp_aggregates[warp_id][yInWarp] = warp_aggregate;
            }
        }
        else if (scanDir == kVerticalReversed)
        {
            if (lane_id < kWarpDimX)
            {
                temp_storage.warp_aggregates[warp_id][xInWarp] = warp_aggregate;
            }
        }
        else
        {
            __trap();
        }

        cub::CTA_SYNC();

        if (scanDir == kHorizontal)
        {
            int firstWarp = (warp_id / kWarpsAcrossBlockDimX) * kWarpsAcrossBlockDimX;
            block_aggregate = temp_storage.warp_aggregates[firstWarp][yInWarp];

            #pragma unroll
            for (int i = 1; i < kWarpsAcrossBlockDimX; ++i)
            {
                int warp = firstWarp + i;

                if (warp_id == warp)
                {
                    warp_prefix = block_aggregate;
                }

                T addend = temp_storage.warp_aggregates[warp][yInWarp];
                block_aggregate = scan_op(block_aggregate, addend);
            }
        }
        else if (scanDir == kVertical)
        {
            int firstWarp = warp_id % kWarpsAcrossBlockDimX;
            block_aggregate = temp_storage.warp_aggregates[firstWarp][xInWarp];

            #pragma unroll
            for (int i = 1; i < kWarpsAcrossBlockDimY; ++i)
            {
                int warp = firstWarp + i * kWarpsAcrossBlockDimX;

                if (warp_id == warp)
                {
                    warp_prefix = block_aggregate;
                }

                T addend = temp_storage.warp_aggregates[warp][xInWarp];
                block_aggregate = scan_op(block_aggregate, addend);
            }
        }
        else if (scanDir == kHorizontalReversed)
        {
            // Suppose warps are tiled as follows in this thread block:
            // 0    1    2
            // 3    4    5
            // 6    7    8
            // lastWarp denotes the id of the warp in the last column in this row
            // e.g, for warps 3, 4, 5, last warp is 5,
            //      for warps 6, 7, 8, last warp is 8.
            int lastWarp = (warp_id / kWarpsAcrossBlockDimX) * kWarpsAcrossBlockDimX + kWarpsAcrossBlockDimX - 1;
            block_aggregate = temp_storage.warp_aggregates[lastWarp][yInWarp];

            #pragma unroll
            for (int i = 1; i < kWarpsAcrossBlockDimX; ++i)
            {
                int warp = lastWarp - i;

                if (warp_id == warp)
                {
                    warp_prefix = block_aggregate;
                }

                T addend = temp_storage.warp_aggregates[warp][yInWarp];
                block_aggregate = scan_op(block_aggregate, addend);
            }
        }
        else if (scanDir == kVerticalReversed)
        {
            int lastWarp = warp_id % kWarpsAcrossBlockDimX + (kWarpsAcrossBlockDimY - 1) * kWarpsAcrossBlockDimX;
            block_aggregate = temp_storage.warp_aggregates[lastWarp][xInWarp];

            #pragma unroll
            for (int i = 1; i < kWarpsAcrossBlockDimY; ++i)
            {
                int warp = lastWarp - i * kWarpsAcrossBlockDimX;

                if (warp_id == warp)
                {
                    warp_prefix = block_aggregate;
                }

                T addend = temp_storage.warp_aggregates[warp][xInWarp];
                block_aggregate = scan_op(block_aggregate, addend);
            }
        }
        else
        {
            __trap();
        }

        // Added to pass compute sanitizer racecheck.
        cub::CTA_SYNC();

        return warp_prefix;
    }

    #if false
    /// Use the warp-wide aggregates and initial-value to compute the calling warp's prefix.  Also returns block-wide aggregate in all threads.
    template <typename ScanOp>
    __device__ __forceinline__ T ComputeWarpPrefix(
            ScanOp scan_op,            ///< [in] Binary scan operator
            T warp_aggregate,     ///< [in] <b>[<em>lane</em><sub>WARP_THREADS - 1</sub> only]</b> Warp-wide aggregate reduction of input items
            T & block_aggregate,   ///< [out] Threadblock-wide aggregate reduction of input items
            const T & initial_value)     ///< [in] Initial value to seed the exclusive scan
    {
        T warp_prefix = ComputeWarpPrefix(scan_op, warp_aggregate, block_aggregate);

        warp_prefix = scan_op(initial_value, warp_prefix);

        if (warp_id == 0)
        {
            warp_prefix = initial_value;
        }

        return warp_prefix;
    }
    #endif  // false



    //---------------------------------------------------------------------
    // Exclusive scans
    //---------------------------------------------------------------------

    #if false
    /// Computes an exclusive thread block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
            T input,              ///< [in] Calling thread's input items
            T & exclusive_output,  ///< [out] Calling thread's output items (may be aliased to \p input)
            const T & initial_value,     ///< [in] Initial value to seed the exclusive scan
            ScanOp scan_op)            ///< [in] Binary scan operator
    {
        T block_aggregate;
        ExclusiveScan(input, exclusive_output, initial_value, scan_op, block_aggregate);
    }


    /// Computes an exclusive thread block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
            T input,              ///< [in] Calling thread's input items
            T & exclusive_output,  ///< [out] Calling thread's output items (may be aliased to \p input)
            const T & initial_value,     ///< [in] Initial value to seed the exclusive scan
            ScanOp scan_op,            ///< [in] Binary scan operator
            T & block_aggregate)   ///< [out] Threadblock-wide aggregate reduction of input items
    {
        // Compute warp scan in each warp.  The exclusive output from each lane0 is invalid.
        T inclusive_output;
        WarpScanT(temp_storage.warp_scan[warp_id]).Scan(input, inclusive_output, exclusive_output, scan_op);

        // Compute the warp-wide prefix and block-wide aggregate for each warp
        T warp_prefix = ComputeWarpPrefix(scan_op, inclusive_output, block_aggregate, initial_value);

        // Apply warp prefix to our lane's partial
        exclusive_output = scan_op(warp_prefix, exclusive_output);
        if (lane_id == 0)
        {
            exclusive_output = warp_prefix;
        }
    }


    /// Computes an exclusive thread block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the thread block's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    template <
            typename ScanOp,
            typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void ExclusiveScan(
            T input,                          ///< [in] Calling thread's input item
            T & exclusive_output,              ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp scan_op,                        ///< [in] Binary scan operator
            BlockPrefixCallbackOp & block_prefix_callback_op)      ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a thread block-wide prefix to be applied to all inputs.
    {
        // Compute block-wide exclusive scan.  The exclusive output from tid0 is invalid.
        T block_aggregate;
        ExclusiveScan(input, exclusive_output, scan_op, block_aggregate);

        // Use the first warp to determine the thread block prefix, returning the result in lane0
        if (warp_id == 0)
        {
            T block_prefix = block_prefix_callback_op(block_aggregate);
            if (lane_id == 0)
            {
                // Share the prefix with all threads
                temp_storage.block_prefix = block_prefix;
                exclusive_output = block_prefix;                // The block prefix is the exclusive output for tid0
            }
        }

        cub::CTA_SYNC();

        // Incorporate thread block prefix into outputs
        T block_prefix = temp_storage.block_prefix;
        if (linear_tid > 0)
        {
            exclusive_output = scan_op(block_prefix, exclusive_output);
        }
    }
    #endif  // false

    //---------------------------------------------------------------------
    // Inclusive scans
    //---------------------------------------------------------------------

    /// Computes an inclusive thread block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.
    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(
            T input,                          ///< [in] Calling thread's input item
            T & inclusive_output,              ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp scan_op,                        ///< [in] Binary scan operator
            ScanDir scanDir)             ///< [in] Scan direction
    {
        T block_aggregate;
        InclusiveScan(input, inclusive_output, scan_op, block_aggregate, scanDir);
    }


    /// Computes an inclusive thread block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(
            T input,                          ///< [in] Calling thread's input item
            T & inclusive_output,              ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp scan_op,                        ///< [in] Binary scan operator
            T & block_aggregate,               ///< [out] Threadblock-wide aggregate reduction of input items
            ScanDir scanDir)             ///< [in] Scan direction
    {
        WarpScanT(temp_storage.warp_scan[warp_id]).InclusiveScan(input, inclusive_output, scan_op, scanDir);

        // Compute the warp-wide prefix and block-wide aggregate for each warp.
        // NEW for ndmamba:
        // Warp prefix for the first-row or first-column warps are ZERO.
        // For {A, B}, zero is invalid initial value!
        // warp_predix and block_aggregate is NOT uniform across the whole warp.
        T warp_prefix = ComputeWarpPrefix(scan_op, inclusive_output, block_aggregate, scanDir);

        // Apply warp prefix to our lane's partial (excluding first-row or first column warps)
        if (scanDir == kHorizontal)
        {
            if (warp_id % kWarpsAcrossBlockDimX != 0)  // exclude first column warps for horizontal scan
            {
                inclusive_output = scan_op(warp_prefix, inclusive_output);
            }
        }
        else if (scanDir == kVertical)
        {
            if (kWarpsAcrossBlockDimX <= warp_id)  // exclude first row warps for vertical scan
            {
                inclusive_output = scan_op(warp_prefix, inclusive_output);
            }
        }
        else if (scanDir == kHorizontalReversed)
        {
            if (warp_id % kWarpsAcrossBlockDimX != kWarpsAcrossBlockDimX - 1)  // exclude last column warps for horizontal scan
            {
                inclusive_output = scan_op(warp_prefix, inclusive_output);
            }
        }
        else if (scanDir == kVerticalReversed)
        {
            if (warp_id < WARPS - kWarpsAcrossBlockDimX)  // exclude last row warps for vertical scan
            {
                inclusive_output = scan_op(warp_prefix, inclusive_output);
            }
        }
        else
        {
            __trap();
        }
    }


    /// Computes an inclusive thread block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the thread block's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
    template <
            typename ScanOp,
            typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void InclusiveScan(
            T input,                          ///< [in] Calling thread's input item
            T & inclusive_output,              ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp scan_op,                        ///< [in] Binary scan operator
            BlockPrefixCallbackOp & block_prefix_callback_op,      ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a thread block-wide prefix to be applied to all inputs.
            ScanDir scanDir)             ///< [in] Scan direction
    {
        T block_aggregate;
        InclusiveScan(input, inclusive_output, scan_op, block_aggregate, scanDir);

        // NOT "Use the first warp to determine the thread block prefix, returning the result in lane0"
        // Horizontal: BLOCK_DIM_Y prefixes; Vertical: BLOCK_DIM_X prefixes.

        auto & tempStorageBlockPrefix = temp_storage.warp_aggregates;

        if (scanDir == kHorizontal || scanDir == kHorizontalReversed)
        {
            if (warp_id % kWarpsAcrossBlockDimX == 0)
            {
                T block_prefix = block_prefix_callback_op(block_aggregate);

                if (lane_id % kWarpDimX == 0)
                {
                    // Share the prefix with all threads
                    tempStorageBlockPrefix[warp_id][lane_id / kWarpDimX] = block_prefix;
                }
            }

            cub::CTA_SYNC();

            // Incorporate thread block prefix into outputs
            int firstWarp = (warp_id / kWarpsAcrossBlockDimX) * kWarpsAcrossBlockDimX;
            T block_prefix = tempStorageBlockPrefix[firstWarp][lane_id / kWarpDimX];
            inclusive_output = scan_op(block_prefix, inclusive_output);
        }
        else if (scanDir == kVertical || scanDir == kVerticalReversed)
        {
            if (warp_id / kWarpsAcrossBlockDimX == 0)
            {
                T block_prefix = block_prefix_callback_op(block_aggregate);

                if (lane_id / kWarpDimX == 0)
                {
                    // Share the prefix with all threads
                    tempStorageBlockPrefix[warp_id][lane_id] = block_prefix;
                }
            }

            cub::CTA_SYNC();

            // Incorporate thread block prefix into outputs
            int firstWarp = warp_id % kWarpsAcrossBlockDimX;
            T block_prefix = tempStorageBlockPrefix[firstWarp][lane_id % kWarpDimX];
            inclusive_output = scan_op(block_prefix, inclusive_output);
        }
        else
        {
            __trap();
        }

        #if false
        if (warp_id == 0)
        {
            T block_prefix = block_prefix_callback_op(block_aggregate);
            if (lane_id == 0)
            {
                // Share the prefix with all threads
                temp_storage.block_prefix = block_prefix;
            }
        }
        #endif  // false
    }


    /// Computes an exclusive thread block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  With no initial value, the output computed for <em>thread</em><sub>0</sub> is undefined.
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
            T input,                          ///< [in] Calling thread's input item
            T & exclusive_output,              ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp scan_op,                        ///< [in] Binary scan operator
            ScanDir scanDir)
    {
        // Compute block-wide exclusive scan.  The exclusive output from tid0 is invalid.
        T block_aggregate;
        ExclusiveScan(input, exclusive_output, scan_op, block_aggregate, scanDir);
    }

    /// Computes an exclusive thread block-wide prefix scan using the specified binary \p scan_op functor.
    /// Each thread contributes one input element.
    /// Also provides every thread with the block-wide \p block_aggregate of all inputs.
    /// \n\n
    /// NEW FOR ndmamba: With no initial value,
    /// the output computed for threads of first-row or first-column
    /// of THIS THERAD BLOCK is invalid.
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
            T input,              ///< [in] Calling thread's input item
            T & exclusive_output,  ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp scan_op,            ///< [in] Binary scan operator
            T & block_aggregate,   ///< [out] Threadblock-wide aggregate reduction of input items
            ScanDir scanDir)
    {
        // Compute warp scan in each warp.  The exclusive output from each lane0 is invalid.
        T inclusive_output;
        WarpScanT(temp_storage.warp_scan[warp_id]).Scan(input, inclusive_output, exclusive_output, scan_op, scanDir);

        // Compute the warp-wide prefix and block-wide aggregate for each warp.
        // NEW for ndmamba: Warp prefixes for first-row or first-column warps are invalid (and are NOT used afterward).
        // For each thread, block_aggregate denotes the INCLUSIVE row/column aggregate of this thread's row/column.
        T warp_prefix = ComputeWarpPrefix(scan_op, inclusive_output, block_aggregate, scanDir);

        // Apply warp prefix to our lane's partial (excluding first-row or first column warps)
        if (scanDir == kHorizontal)
        {
            if (warp_id % kWarpsAcrossBlockDimX != 0)  // exclude first column warps for horizontal scan
            {
                exclusive_output = scan_op(warp_prefix, exclusive_output);
            }

            // exclusive_output for first columns in all warps are identical to second-columns
            // (shuffle_up does not modify lane 0 elements).
            // Thus, we need to overwrite exclusive_ouput s for first-column lanes in each warp.
            // NOTE THAT, warp_prefix is invalid for all warps in first column.
            // So this statement is meaningless for first-column lanes in first-column warps,
            // but meaningful for first-column lanes in all other warps.
            if (lane_id % kWarpDimX == 0)
            {
                exclusive_output = warp_prefix;
            }
        }
        else if (scanDir == kVertical)
        {
            if (kWarpsAcrossBlockDimX <= warp_id)  // exclude first row warps for vertical scan
            {
                exclusive_output = scan_op(warp_prefix, exclusive_output);
            }

            // Overwrite exclusive_ouput s for first-row lanes in each warp.
            if (lane_id < kWarpDimX)
            {
                exclusive_output = warp_prefix;
            }
        }
        else if (scanDir == kHorizontalReversed)
        {
            if (warp_id % kWarpsAcrossBlockDimX != kWarpsAcrossBlockDimX - 1)  // exclude last column warps for horizontal scan
            {
                exclusive_output = scan_op(warp_prefix, exclusive_output);
            }

            if (lane_id % kWarpDimX == kWarpDimX - 1)
            {
                exclusive_output = warp_prefix;
            }
        }
        else if (scanDir == kVerticalReversed)
        {
            if (warp_id < WARPS - kWarpsAcrossBlockDimX)  // exclude last row warps for vertical scan
            {
                exclusive_output = scan_op(warp_prefix, exclusive_output);
            }

            if (WARP_THREADS - kWarpDimX <= lane_id)
            {
                exclusive_output = warp_prefix;
            }
        }
        else
        {
            __trap();
        }
    }


    /// Computes an exclusive thread block-wide prefix scan using the specified binary \p scan_op functor.
    /// Each thread contributes one input element.
    /// the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block,
    /// and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value
    /// that logically prefixes the thread block's scan inputs.
    /// Also provides every thread with the block-wide \p block_aggregate of all inputs.
    /// NEW for ndmamba:
    /// For forward scans, prefix callbacks for the leftmost col (top-most row) is used (and populated);
    /// For backward scans, prefix callbacks for the rightmost col (bottom-most row) is used (and polulated).
    template <
            typename ScanOp,
            typename BlockPrefixCallbackOp>
    __device__ __forceinline__ void ExclusiveScan(
            T input,                          ///< [in] Calling thread's input item
            T & exclusive_output,              ///< [out] Calling thread's output item (may be aliased to \p input)
            ScanOp scan_op,                        ///< [in] Binary scan operator
            BlockPrefixCallbackOp & block_prefix_callback_op,      ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b> Call-back functor for specifying a thread block-wide prefix to be applied to all inputs.
            ScanDir scanDir)
    {
        // Compute block-wide exclusive scan.  The exclusive output from tid0 is invalid.
        // For each thread, block_aggregate denotes the INCLUSIVE row/column aggregate of this thread's row/column.
        // (Although it's populated by ExclusiveScan, inside it's computed with warpScan's inclusive_output).
        T block_aggregate;
        ExclusiveScan(input, exclusive_output, scan_op, block_aggregate, scanDir);

        auto & tempStorageBlockPrefix = temp_storage.warp_aggregates;

        if (scanDir == kHorizontal)
        {
            if (warp_id % kWarpsAcrossBlockDimX == 0)
            {
                T block_prefix = block_prefix_callback_op(block_aggregate);

                if (lane_id % kWarpDimX == 0)
                {
                    // Share the prefix with all threads
                    tempStorageBlockPrefix[warp_id][lane_id / kWarpDimX] = block_prefix;
                    exclusive_output = block_prefix;  // The block prefix is the exclusive output for the first column
                }
            }

            cub::CTA_SYNC();

            // Incorporate thread block prefix into outputs
            int firstWarp = (warp_id / kWarpsAcrossBlockDimX) * kWarpsAcrossBlockDimX;
            T block_prefix = tempStorageBlockPrefix[firstWarp][lane_id / kWarpDimX];

            if (warp_id % kWarpsAcrossBlockDimX != 0 || lane_id % kWarpDimX != 0)
            {
                exclusive_output = scan_op(block_prefix, exclusive_output);
            }
        }
        else if (scanDir == kVertical)
        {
            if (warp_id / kWarpsAcrossBlockDimX == 0)
            {
                T block_prefix = block_prefix_callback_op(block_aggregate);

                if (lane_id / kWarpDimX == 0)
                {
                    // Share the prefix with all threads
                    tempStorageBlockPrefix[warp_id][lane_id] = block_prefix;
                    exclusive_output = block_prefix;  // The block prefix is the exclusive output for the first row
                }
            }

            cub::CTA_SYNC();

            // Incorporate thread block prefix into outputs
            int firstWarp = warp_id % kWarpsAcrossBlockDimX;
            T block_prefix = tempStorageBlockPrefix[firstWarp][lane_id % kWarpDimX];

            if (warp_id / kWarpsAcrossBlockDimX != 0 || lane_id / kWarpDimX != 0)
            {
                exclusive_output = scan_op(block_prefix, exclusive_output);
            }
        }
        else if (scanDir == kHorizontalReversed)
        {
            if (warp_id % kWarpsAcrossBlockDimX == kWarpsAcrossBlockDimX - 1)
            {
                // Suppose warps are tiled as follows in this thread block:
                // 0    1
                // 2    3
                // 4    5
                // 6    7
                // block_prefix_callback_op.running_prefix will be correct for
                // warps 1, 3, 5, and 7 (rightmost-column warps).
                T block_prefix = block_prefix_callback_op(block_aggregate);

                if (lane_id % kWarpDimX == kWarpDimX - 1)
                {
                    // Share the prefix with all threads
                    tempStorageBlockPrefix[warp_id][lane_id / kWarpDimX] = block_prefix;
                    exclusive_output = block_prefix;  // The block prefix is the exclusive output for the first column
                }
            }

            cub::CTA_SYNC();

            // Incorporate thread block prefix into outputs
            int lastWarp = (warp_id / kWarpsAcrossBlockDimX) * kWarpsAcrossBlockDimX + kWarpsAcrossBlockDimX - 1;
            T block_prefix = tempStorageBlockPrefix[lastWarp][lane_id / kWarpDimX];

            // Exclude threads in the rightmost column of this thread block:
            if (warp_id % kWarpsAcrossBlockDimX != kWarpsAcrossBlockDimX - 1 || lane_id % kWarpDimX != kWarpDimX - 1)
            {
                exclusive_output = scan_op(block_prefix, exclusive_output);
            }
        }
        else if (scanDir == kVerticalReversed)
        {
            if (WARPS - kWarpsAcrossBlockDimX <= warp_id)
            {
                T block_prefix = block_prefix_callback_op(block_aggregate);

                if (WARP_THREADS - kWarpDimX <= lane_id)
                {
                    // Share the prefix with all threads
                    tempStorageBlockPrefix[warp_id][lane_id % kWarpDimX] = block_prefix;
                    exclusive_output = block_prefix;  // The block prefix is the exclusive output for the first row
                }
            }

            cub::CTA_SYNC();

            // Incorporate thread block prefix into outputs
            int lastWarp = warp_id % kWarpsAcrossBlockDimX + (kWarpsAcrossBlockDimY - 1) * kWarpsAcrossBlockDimX;
            T block_prefix = tempStorageBlockPrefix[lastWarp][lane_id % kWarpDimX];

            if (warp_id < WARPS - kWarpsAcrossBlockDimX || lane_id < WARP_THREADS - kWarpDimX)
            {
                exclusive_output = scan_op(block_prefix, exclusive_output);
            }
        }
        else
        {
            __trap();
        }

        #if false
        // Use the first warp to determine the thread block prefix, returning the result in lane0
        if (warp_id == 0)
        {
            T block_prefix = block_prefix_callback_op(block_aggregate);

            if (lane_id == 0)
            {
                // Share the prefix with all threads
                temp_storage.block_prefix = block_prefix;
                exclusive_output = block_prefix;                // The block prefix is the exclusive output for tid0
            }
        }

        cub::CTA_SYNC();

        // Incorporate thread block prefix into outputs
        T block_prefix = temp_storage.block_prefix;
        if (linear_tid > 0)
        {
            exclusive_output = scan_op(block_prefix, exclusive_output);
        }
        #endif  // false
    }
};

}  // namespace ndmamba

