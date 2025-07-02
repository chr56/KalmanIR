#pragma once

#include <cub/config.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_type.cuh>
#include <cub/util_ptx.cuh>


namespace ndmamba
{

template <
        typename T,                      ///< Data type being scanned
        int LOGICAL_WARP_THREADS,   ///< Number of threads per logical warp
        int PTX_ARCH,               ///< The PTX compute capability for which to to specialize this collective
        int kSegLen
>
struct SegWarpScanShfl
{
    //---------------------------------------------------------------------
    // Constants and type definitions
    //---------------------------------------------------------------------

    enum
    {
        /// Whether the logical warp size and the PTX warp size coincide
        IS_ARCH_WARP = (LOGICAL_WARP_THREADS == CUB_WARP_THREADS(PTX_ARCH)),

        /// The number of warp scan steps for horizontal scan
        kHorizontalSteps = cub::Log2<kSegLen>::VALUE,

        /// The number of warp scan steps for vertical scan
        kVerticalSteps = cub::Log2<LOGICAL_WARP_THREADS>::VALUE,

        /// The 5-bit SHFL mask for logically splitting warps into sub-segments starts 8-bits up
        SHFL_C = (CUB_WARP_THREADS(PTX_ARCH) - LOGICAL_WARP_THREADS) << 8
    };

    template <typename S>
    struct IntegerTraits
    {
        enum
        {
            ///Whether the data type is a small (32b or less) integer for which we can use a single SFHL instruction per exchange
            IS_SMALL_UNSIGNED =
            (cub::Traits<S>::CATEGORY == cub::UNSIGNED_INTEGER) && (sizeof(S) <= sizeof(unsigned int))
        };
    };

    /// Shared memory storage layout type
    struct TempStorage
    {
    };


    //---------------------------------------------------------------------
    // Thread fields
    //---------------------------------------------------------------------

    /// Lane index in logical warp
    unsigned int lane_id;

    /// Logical warp index in 32-thread physical warp
    unsigned int warp_id;

    /// 32-thread physical warp member mask of logical warp
    unsigned int member_mask;

    //---------------------------------------------------------------------
    // Construction
    //---------------------------------------------------------------------

    /// Constructor
    explicit __device__ __forceinline__
    SegWarpScanShfl(TempStorage & /*temp_storage*/)
            : lane_id(cub::LaneId()), warp_id(IS_ARCH_WARP ? 0 : (lane_id / LOGICAL_WARP_THREADS)),
              member_mask(cub::WarpMask<LOGICAL_WARP_THREADS, PTX_ARCH>(warp_id))
    {
        static_assert(IS_ARCH_WARP && (LOGICAL_WARP_THREADS % kSegLen == 0));

        #if false
        if (!IS_ARCH_WARP)
        {
            lane_id = lane_id % LOGICAL_WARP_THREADS;
        }
        #endif  // false
    }


    //---------------------------------------------------------------------
    // Inclusive scan steps
    //---------------------------------------------------------------------

    #if false
    /// Inclusive prefix scan step (specialized for summation across int32 types)
    __device__ __forceinline__ int InclusiveScanStep(
            int input,              ///< [in] Calling thread's input item.
            cub::Sum        /*scan_op*/,        ///< [in] Binary scan operator
            int first_lane,         ///< [in] Index of first lane in segment
            int offset)             ///< [in] Up-offset to pull from
    {
        int output;
        int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

        // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
        asm volatile(
                "{"
                "  .reg .s32 r0;"
                "  .reg .pred p;"
                "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
                "  @p add.s32 r0, r0, %4;"
                "  mov.s32 %0, r0;"
                "}"
                : "=r"(output) : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));
#else
        asm volatile(
            "{"
            "  .reg .s32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.s32 r0, r0, %4;"
            "  mov.s32 %0, r0;"
            "}"
            : "=r"(output) : "r"(input), "r"(offset), "r"(shfl_c), "r"(input));
#endif

        return output;
    }

    /// Inclusive prefix scan step (specialized for summation across uint32 types)
    __device__ __forceinline__ unsigned int InclusiveScanStep(
            unsigned int input,              ///< [in] Calling thread's input item.
            cub::Sum        /*scan_op*/,        ///< [in] Binary scan operator
            int first_lane,         ///< [in] Index of first lane in segment
            int offset)             ///< [in] Up-offset to pull from
    {
        unsigned int output;
        int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

        // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
        asm volatile(
                "{"
                "  .reg .u32 r0;"
                "  .reg .pred p;"
                "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
                "  @p add.u32 r0, r0, %4;"
                "  mov.u32 %0, r0;"
                "}"
                : "=r"(output) : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));
#else
        asm volatile(
            "{"
            "  .reg .u32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.u32 r0, r0, %4;"
            "  mov.u32 %0, r0;"
            "}"
            : "=r"(output) : "r"(input), "r"(offset), "r"(shfl_c), "r"(input));
#endif

        return output;
    }


    /// Inclusive prefix scan step (specialized for summation across fp32 types)
    __device__ __forceinline__ float InclusiveScanStep(
            float input,              ///< [in] Calling thread's input item.
            cub::Sum        /*scan_op*/,        ///< [in] Binary scan operator
            int first_lane,         ///< [in] Index of first lane in segment
            int offset)             ///< [in] Up-offset to pull from
    {
        float output;
        int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

        // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
        asm volatile(
                "{"
                "  .reg .f32 r0;"
                "  .reg .pred p;"
                "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
                "  @p add.f32 r0, r0, %4;"
                "  mov.f32 %0, r0;"
                "}"
                : "=f"(output) : "f"(input), "r"(offset), "r"(shfl_c), "f"(input), "r"(member_mask));
#else
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(output) : "f"(input), "r"(offset), "r"(shfl_c), "f"(input));
#endif

        return output;
    }


    /// Inclusive prefix scan step (specialized for summation across unsigned long long types)
    __device__ __forceinline__ unsigned long long InclusiveScanStep(
            unsigned long long input,              ///< [in] Calling thread's input item.
            cub::Sum            /*scan_op*/,        ///< [in] Binary scan operator
            int first_lane,         ///< [in] Index of first lane in segment
            int offset)             ///< [in] Up-offset to pull from
    {
        unsigned long long output;
        int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

        // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
        asm volatile(
                "{"
                "  .reg .u64 r0;"
                "  .reg .u32 lo;"
                "  .reg .u32 hi;"
                "  .reg .pred p;"
                "  mov.b64 {lo, hi}, %1;"
                "  shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
                "  shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
                "  mov.b64 r0, {lo, hi};"
                "  @p add.u64 r0, r0, %4;"
                "  mov.u64 %0, r0;"
                "}"
                : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c), "l"(input), "r"(member_mask));
#else
        asm volatile(
            "{"
            "  .reg .u64 r0;"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 r0, {lo, hi};"
            "  @p add.u64 r0, r0, %4;"
            "  mov.u64 %0, r0;"
            "}"
            : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c), "l"(input));
#endif

        return output;
    }


    /// Inclusive prefix scan step (specialized for summation across long long types)
    __device__ __forceinline__ long long InclusiveScanStep(
            long long input,              ///< [in] Calling thread's input item.
            cub::Sum        /*scan_op*/,        ///< [in] Binary scan operator
            int first_lane,         ///< [in] Index of first lane in segment
            int offset)             ///< [in] Up-offset to pull from
    {
        long long output;
        int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

        // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
        asm volatile(
                "{"
                "  .reg .s64 r0;"
                "  .reg .u32 lo;"
                "  .reg .u32 hi;"
                "  .reg .pred p;"
                "  mov.b64 {lo, hi}, %1;"
                "  shfl.sync.up.b32 lo|p, lo, %2, %3, %5;"
                "  shfl.sync.up.b32 hi|p, hi, %2, %3, %5;"
                "  mov.b64 r0, {lo, hi};"
                "  @p add.s64 r0, r0, %4;"
                "  mov.s64 %0, r0;"
                "}"
                : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c), "l"(input), "r"(member_mask));
#else
        asm volatile(
            "{"
            "  .reg .s64 r0;"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 r0, {lo, hi};"
            "  @p add.s64 r0, r0, %4;"
            "  mov.s64 %0, r0;"
            "}"
            : "=l"(output) : "l"(input), "r"(offset), "r"(shfl_c), "l"(input));
#endif

        return output;
    }


    /// Inclusive prefix scan step (specialized for summation across fp64 types)
    __device__ __forceinline__ double InclusiveScanStep(
            double input,              ///< [in] Calling thread's input item.
            cub::Sum        /*scan_op*/,        ///< [in] Binary scan operator
            int first_lane,         ///< [in] Index of first lane in segment
            int offset)             ///< [in] Up-offset to pull from
    {
        double output;
        int shfl_c = first_lane | SHFL_C;   // Shuffle control (mask and first-lane)

        // Use predicate set from SHFL to guard against invalid peers
#ifdef CUB_USE_COOPERATIVE_GROUPS
        asm volatile(
                "{"
                "  .reg .u32 lo;"
                "  .reg .u32 hi;"
                "  .reg .pred p;"
                "  .reg .f64 r0;"
                "  mov.b64 %0, %1;"
                "  mov.b64 {lo, hi}, %1;"
                "  shfl.sync.up.b32 lo|p, lo, %2, %3, %4;"
                "  shfl.sync.up.b32 hi|p, hi, %2, %3, %4;"
                "  mov.b64 r0, {lo, hi};"
                "  @p add.f64 %0, %0, r0;"
                "}"
                : "=d"(output) : "d"(input), "r"(offset), "r"(shfl_c), "r"(member_mask));
#else
        asm volatile(
            "{"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  .reg .f64 r0;"
            "  mov.b64 %0, %1;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 r0, {lo, hi};"
            "  @p add.f64 %0, %0, r0;"
            "}"
            : "=d"(output) : "d"(input), "r"(offset), "r"(shfl_c));
#endif

        return output;
    }


    /*
        /// Inclusive prefix scan (specialized for ReduceBySegmentOp<cub::Sum> across KeyValuePair<OffsetT, Value> types)
        template <typename Value, typename OffsetT>
        __device__ __forceinline__ KeyValuePair<OffsetT, Value>InclusiveScanStep(
            KeyValuePair<OffsetT, Value>    input,              ///< [in] Calling thread's input item.
            ReduceBySegmentOp<cub::Sum>     scan_op,            ///< [in] Binary scan operator
            int                             first_lane,         ///< [in] Index of first lane in segment
            int                             offset)             ///< [in] Up-offset to pull from
        {
            KeyValuePair<OffsetT, Value> output;

            output.value = InclusiveScanStep(input.value, cub::Sum(), first_lane, offset, Int2Type<IntegerTraits<Value>::IS_SMALL_UNSIGNED>());
            output.key = InclusiveScanStep(input.key, cub::Sum(), first_lane, offset, Int2Type<IntegerTraits<OffsetT>::IS_SMALL_UNSIGNED>());

            if (input.key > 0)
                output.value = input.value;

            return output;
        }
    */
    #endif  // false

    /// Inclusive prefix scan step (generic)
    template <typename InOutType, typename ScanOpT>
    __device__ __forceinline__ InOutType InclusiveScanStep(
            InOutType input,              ///< [in] Calling thread's input item.
            ScanOpT scan_op,            ///< [in] Binary scan operator
            int first_or_last_lane,         ///< [in] Index of first lane in segment
            int offset,             ///< [in] Up-offset to pull from
            ScanDir scanDir)
    {
        if (scanDir <= kVertical)
        {
            InOutType temp = cub::ShuffleUp<LOGICAL_WARP_THREADS>(input, offset, first_or_last_lane, member_mask);
            InOutType output = scan_op(temp, input);

            if (static_cast<int>(lane_id) < first_or_last_lane + offset)
            {
                output = input;
            }

            return output;
        }
        else if (scanDir <= kVerticalReversed)
        {
            InOutType temp = cub::ShuffleDown<LOGICAL_WARP_THREADS>(input, offset, first_or_last_lane, member_mask);
            InOutType output = scan_op(temp, input);

            if (first_or_last_lane - offset < static_cast<int>(lane_id))
            {
                output = input;
            }

            return output;
        }
        else
        {
            __trap();
        }
    }

    #if false
    /// Inclusive prefix scan step (specialized for small integers size 32b or less)
    template <typename InOutType, typename ScanOpT>
    __device__ __forceinline__ InOutType InclusiveScanStep(
            InOutType input,              ///< [in] Calling thread's input item.
            ScanOpT scan_op,            ///< [in] Binary scan operator
            int first_lane,         ///< [in] Index of first lane in segment
            int offset,             ///< [in] Up-offset to pull from
            cub::Int2Type<true>  /*is_small_unsigned*/)  ///< [in] Marker type indicating whether T is a small integer
    {
        return InclusiveScanStep(input, scan_op, first_lane, offset);
    }


    /// Inclusive prefix scan step (specialized for types other than small integers size 32b or less)
    template <typename InOutType, typename ScanOpT>
    __device__ __forceinline__ InOutType InclusiveScanStep(
            InOutType input,              ///< [in] Calling thread's input item.
            ScanOpT scan_op,            ///< [in] Binary scan operator
            int first_lane,         ///< [in] Index of first lane in segment
            int offset,             ///< [in] Up-offset to pull from
            cub::Int2Type<false> /*is_small_unsigned*/)  ///< [in] Marker type indicating whether T is a small integer
    {
        return InclusiveScanStep(input, scan_op, first_lane, offset);
    }
    #endif  // false

    /******************************************************************************
     * Interface
     ******************************************************************************/

    //---------------------------------------------------------------------
    // Broadcast
    //---------------------------------------------------------------------

    /// Broadcast
    __device__ __forceinline__ T Broadcast(
            T input,              ///< [in] The value to broadcast
            int src_lane)           ///< [in] Which warp lane is to do the broadcasting
    {
        return cub::ShuffleIndex<LOGICAL_WARP_THREADS>(input, src_lane, member_mask);
    }


    //---------------------------------------------------------------------
    // Inclusive operations
    //---------------------------------------------------------------------

    /// Inclusive scan
    template <typename InOutType, typename ScanOpT>
    __device__ __forceinline__ void InclusiveScan(
            InOutType input,              ///< [in] Calling thread's input item.
            InOutType & inclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOpT scan_op,            ///< [in] Binary scan operator
            ScanDir scanDir)       ///< [in] Scan dir
    {
        inclusive_output = input;

        int segment_first_or_last_lane = 0;
        int step = 0;
        int stepMax = 0;

        if (scanDir == kHorizontal)
        {
            segment_first_or_last_lane = (lane_id >> kHorizontalSteps) << kHorizontalSteps;
            stepMax = kHorizontalSteps;
        }
        else if (scanDir == kVertical)
        {
            segment_first_or_last_lane = 0;
            step = kHorizontalSteps;
            stepMax = kVerticalSteps;
        }
        else if (scanDir == kHorizontalReversed)
        {
            segment_first_or_last_lane = ((lane_id >> kHorizontalSteps) << kHorizontalSteps) + kSegLen - 1;
            stepMax = kHorizontalSteps;
        }
        else if (scanDir == kVerticalReversed)
        {
            segment_first_or_last_lane = LOGICAL_WARP_THREADS - 1;
            step = kHorizontalSteps;
            stepMax = kVerticalSteps;
        }
        else
        {
            // TODO: scanDir
            __trap();
        }

        // Iterate scan steps
        #pragma unroll
        for ( ; step < stepMax; step++)
        {
            inclusive_output = InclusiveScanStep(
                    inclusive_output,
                    scan_op,
                    segment_first_or_last_lane,
                    (1 << step),
                    scanDir
            );
        }
    }

    #if false
    /// Inclusive scan, specialized for reduce-value-by-key
    template <typename KeyT, typename ValueT, typename ReductionOpT>
    __device__ __forceinline__ void InclusiveScan(
            cub::KeyValuePair<KeyT, ValueT> input,              ///< [in] Calling thread's input item.
            cub::KeyValuePair<KeyT, ValueT> & inclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
            cub::ReduceByKeyOp<ReductionOpT> scan_op)            ///< [in] Binary scan operator
    {
        inclusive_output = input;

        KeyT pred_key = cub::ShuffleUp<LOGICAL_WARP_THREADS>(inclusive_output.key, 1, 0, member_mask);

        unsigned int ballot = WARP_BALLOT((pred_key != inclusive_output.key), member_mask);

        // Mask away all lanes greater than ours
        ballot = ballot & cub::LaneMaskLe();

        // Find index of first set bit
        int segment_first_lane = CUB_MAX(0, 31 - __clz(ballot));

        // Iterate scan steps
        #pragma unroll
        for (int STEP = 0; STEP < kSteps; STEP++)
        {
            inclusive_output.value = InclusiveScanStep(
                    inclusive_output.value,
                    scan_op.op,
                    segment_first_lane,
                    (1 << STEP),
                    cub::Int2Type<IntegerTraits<T>::IS_SMALL_UNSIGNED>());
        }
    }
    #endif  // false


    /// Inclusive scan with aggregate
    template <typename ScanOpT>
    __device__ __forceinline__ void InclusiveScan(
            T input,              ///< [in] Calling thread's input item.
            T & inclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOpT scan_op,            ///< [in] Binary scan operator
            T & warp_aggregate,    ///< [out] Warp-wide aggregate reduction of input items.
            ScanDir scanDir)       ///< [in] Scan dir
    {
        InclusiveScan(input, inclusive_output, scan_op, scanDir);

        int src_lane;

        if (scanDir == kHorizontal)
        {
            src_lane = ((lane_id >> kHorizontalSteps) << kHorizontalSteps) + kSegLen - 1;
        }
        else if (scanDir == kVertical)
        {
            src_lane = (lane_id % kSegLen) + LOGICAL_WARP_THREADS - kSegLen;
        }
        else if (scanDir == kHorizontalReversed)
        {
            src_lane = ((lane_id >> kHorizontalSteps) << kHorizontalSteps);
        }
        else if (scanDir == kVerticalReversed)
        {
            src_lane = lane_id % kSegLen;
        }
        else
        {
            __trap();
        }

        // Grab aggregate from last warp lane
        warp_aggregate = cub::ShuffleIndex<LOGICAL_WARP_THREADS>(
                inclusive_output,
                src_lane,
                member_mask
        );
    }

    //---------------------------------------------------------------------
    // Get exclusive from inclusive
    //---------------------------------------------------------------------

    /// Update inclusive and exclusive using input and inclusive
    template <typename ScanOpT>
    __device__ __forceinline__ void Update(
            T                       /*input*/,          ///< [in]
            T & inclusive,         ///< [in, out]
            T & exclusive,         ///< [out]
            ScanOpT                 /*scan_op*/,        ///< [in]
            ScanDir scanDir)     ///< [in]
    {
        if (scanDir == kHorizontal)
        {
            int src_offset = 1;
            int first_thread = (lane_id >> kHorizontalSteps) << kHorizontalSteps;

            // initial value unknown
            exclusive = cub::ShuffleUp<LOGICAL_WARP_THREADS>(inclusive,
                                                             src_offset,
                                                             first_thread,
                                                             member_mask);
        }
        else if (scanDir == kVertical)
        {
            int src_offset = kSegLen;
            int first_thread = 0;

            // initial value unknown
            exclusive = cub::ShuffleUp<LOGICAL_WARP_THREADS>(inclusive,
                                                             src_offset,
                                                             first_thread,
                                                             member_mask);
        }
        else if (scanDir == kHorizontalReversed)
        {
            int src_offset = 1;
            int last_thread = ((lane_id >> kHorizontalSteps) << kHorizontalSteps) + kSegLen - 1;

            // initial value unknown
            exclusive = cub::ShuffleDown<LOGICAL_WARP_THREADS>(inclusive,
                                                               src_offset,
                                                               last_thread,
                                                               member_mask);
        }
        else if (scanDir == kVerticalReversed)
        {
            int src_offset = kSegLen;
            int last_thread = LOGICAL_WARP_THREADS - 1;

            // initial value unknown
            exclusive = cub::ShuffleDown<LOGICAL_WARP_THREADS>(inclusive,
                                                               src_offset,
                                                               last_thread,
                                                               member_mask);
        }
        else
        {
            __trap();
        }
    }

    #if false
    /// Update inclusive and exclusive using initial value using input, inclusive, and initial value
    template <typename ScanOpT>
    __device__ __forceinline__ void Update(
            T                       /*input*/,
            T & inclusive,
            T & exclusive,
            ScanOpT scan_op,
            T initial_value,
            ScanDir scanDir)
    {
        inclusive = scan_op(initial_value, inclusive);

        exclusive = cub::ShuffleUp<LOGICAL_WARP_THREADS>(inclusive, 1, 0, member_mask);

        if (lane_id == 0)
        {
            exclusive = initial_value;
        }
    }


    /// Update inclusive, exclusive, and warp aggregate using input and inclusive
    template <typename ScanOpT>
    __device__ __forceinline__ void Update(
            T input,
            T & inclusive,
            T & exclusive,
            T & warp_aggregate,
            ScanOpT scan_op,
            ScanDir scanDir)
    {
        warp_aggregate = cub::ShuffleIndex<LOGICAL_WARP_THREADS>(inclusive, LOGICAL_WARP_THREADS - 1, member_mask);
        Update(input, inclusive, exclusive, scan_op, is_integer);
    }

    /// Update inclusive, exclusive, and warp aggregate using input, inclusive, and initial value
    template <typename ScanOpT>
    __device__ __forceinline__ void Update(
            T input,
            T & inclusive,
            T & exclusive,
            T & warp_aggregate,
            ScanOpT scan_op,
            T initial_value,
            ScanDir scanDir)
    {
        warp_aggregate = cub::ShuffleIndex<LOGICAL_WARP_THREADS>(inclusive, LOGICAL_WARP_THREADS - 1, member_mask);
        Update(input, inclusive, exclusive, scan_op, initial_value, is_integer);
    }
    #endif  // false
};

}  // namespace ndmamba