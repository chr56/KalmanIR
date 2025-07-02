#pragma once

#include <cub/config.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_type.cuh>

#include "scan/commons.h"
#include "scan/warp_scan_shfl.cuh"


namespace ndmamba
{

template <
        typename T,
        int kSegLen,
        int LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS,
        int PTX_ARCH = CUB_PTX_ARCH>
class SegWarpScan
{
private:

    /******************************************************************************
     * Constants and type definitions
     ******************************************************************************/

    enum
    {
        /// Whether the logical warp size and the PTX warp size coincide
        IS_ARCH_WARP = (LOGICAL_WARP_THREADS == CUB_WARP_THREADS(PTX_ARCH)),

        /// Whether the logical warp size is a power-of-two
        IS_POW_OF_TWO = ((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)) == 0),

//        /// Whether the data type is an integer (which has fully-associative addition)
//        IS_INTEGER = ((cub::Traits<T>::CATEGORY == cub::SIGNED_INTEGER) ||
//                      (cub::Traits<T>::CATEGORY == cub::UNSIGNED_INTEGER))
    };

    /// Internal specialization.  Use SHFL-based scan if (architecture is >= SM30) and (LOGICAL_WARP_THREADS is a power-of-two)
    typedef SegWarpScanShfl<T, LOGICAL_WARP_THREADS, PTX_ARCH, kSegLen> InternalWarpScan;

    /// Shared memory storage layout type for WarpScan
    typedef typename InternalWarpScan::TempStorage InternalWarpSacnTempStorage;


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    InternalWarpSacnTempStorage & temp_storage;

    unsigned int lane_id;



    /******************************************************************************
     * Public types
     ******************************************************************************/

public:

    /// \smemstorage{WarpScan}
    struct TempStorage : cub::Uninitialized<InternalWarpSacnTempStorage>
    {
    };


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.  Logical warp and lane identifiers are constructed from <tt>threadIdx.x</tt>.
     */
    __device__ __forceinline__ SegWarpScan(
            TempStorage & temp_storage)             ///< [in] Reference to memory allocation having layout type TempStorage
            :
            temp_storage(temp_storage.Alias()),
            lane_id(IS_ARCH_WARP ?
                    cub::LaneId() :
                    cub::LaneId() % LOGICAL_WARP_THREADS)
    {
    }

    /******************************************************************//**
     * \name Inclusive prefix scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes an inclusive prefix scan using the specified binary scan functor across the calling warp.
     *
     * \par
     *  - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates four concurrent warp-wide inclusive prefix max scans within a block of
     * 128 threads (one per each of the 32-thread warps).
     * \par
     * \code
     * #include <cub/cub.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize WarpScan for type int
     *     typedef cub::WarpScan<int> WarpScan;
     *
     *     // Allocate WarpScan shared memory for 4 warps
     *     __shared__ typename WarpScan::TempStorage temp_storage[4];
     *
     *     // Obtain one input item per thread
     *     int thread_data = ...
     *
     *     // Compute inclusive warp-wide prefix max scans
     *     int warp_id = threadIdx.x / 32;
     *     WarpScan(temp_storage[warp_id]).InclusiveScan(thread_data, thread_data, cub::Max());
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{0, -1, 2, -3, ..., 126, -127}</tt>.
     * The corresponding output \p thread_data in the first warp would be
     * <tt>0, 0, 2, 2, ..., 30, 30</tt>, the output for the second warp would be <tt>32, 32, 34, 34, ..., 62, 62</tt>, etc.
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(
            T input,              ///< [in] Calling thread's input item.
            T & inclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp scan_op,             ///< [in] Binary scan operator
            ScanDir scanDir)       ///< [in] Scan dir
    {
        InternalWarpScan(temp_storage).InclusiveScan(input, inclusive_output, scan_op, scanDir);
    }


    /**
     * \brief Computes an inclusive prefix scan using the specified binary scan functor across the calling warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * \par
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates four concurrent warp-wide inclusive prefix max scans within a block of
     * 128 threads (one per each of the 32-thread warps).
     * \par
     * \code
     * #include <cub/cub.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize WarpScan for type int
     *     typedef cub::WarpScan<int> WarpScan;
     *
     *     // Allocate WarpScan shared memory for 4 warps
     *     __shared__ typename WarpScan::TempStorage temp_storage[4];
     *
     *     // Obtain one input item per thread
     *     int thread_data = ...
     *
     *     // Compute inclusive warp-wide prefix max scans
     *     int warp_aggregate;
     *     int warp_id = threadIdx.x / 32;
     *     WarpScan(temp_storage[warp_id]).InclusiveScan(
     *         thread_data, thread_data, cub::Max(), warp_aggregate);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{0, -1, 2, -3, ..., 126, -127}</tt>.
     * The corresponding output \p thread_data in the first warp would be
     * <tt>0, 0, 2, 2, ..., 30, 30</tt>, the output for the second warp would be <tt>32, 32, 34, 34, ..., 62, 62</tt>, etc.
     * Furthermore, \p warp_aggregate would be assigned \p 30 for threads in the first warp, \p 62 for threads
     * in the second warp, etc.
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(
            T input,              ///< [in] Calling thread's input item.
            T & inclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp scan_op,            ///< [in] Binary scan operator
            T & seg_aggregate,    ///< [out] Segment warp-wide aggregate reduction of input items.
            ScanDir scanDir)       ///< [in] Scan dir
    {
        InternalWarpScan(temp_storage).InclusiveScan(input, inclusive_output, scan_op, seg_aggregate, scanDir);
    }


    //@}  end member group

    #if false
    /******************************************************************//**
     * \name Exclusive prefix scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor across the calling warp.  Because no initial value is supplied, the \p output computed for <em>warp-lane</em><sub>0</sub> is undefined.
     *
     * \par
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates four concurrent warp-wide exclusive prefix max scans within a block of
     * 128 threads (one per each of the 32-thread warps).
     * \par
     * \code
     * #include <cub/cub.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize WarpScan for type int
     *     typedef cub::WarpScan<int> WarpScan;
     *
     *     // Allocate WarpScan shared memory for 4 warps
     *     __shared__ typename WarpScan::TempStorage temp_storage[4];
     *
     *     // Obtain one input item per thread
     *     int thread_data = ...
     *
     *     // Compute exclusive warp-wide prefix max scans
     *     int warp_id = threadIdx.x / 32;
     *     WarpScan(temp_storage[warp_id]).ExclusiveScan(thread_data, thread_data, cub::Max());
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{0, -1, 2, -3, ..., 126, -127}</tt>.
     * The corresponding output \p thread_data in the first warp would be
     * <tt>?, 0, 0, 2, ..., 28, 30</tt>, the output for the second warp would be <tt>?, 32, 32, 34, ..., 60, 62</tt>, etc.
     * (The output \p thread_data in warp lane<sub>0</sub> is undefined.)
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
            T input,              ///< [in] Calling thread's input item.
            T & exclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp scan_op)            ///< [in] Binary scan operator
    {
        InternalWarpScan internal(temp_storage);

        T inclusive_output;
        internal.InclusiveScan(input, inclusive_output, scan_op);

        internal.Update(
                input,
                inclusive_output,
                exclusive_output,
                scan_op,
                cub::Int2Type<IS_INTEGER>());
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor across the calling warp.
     *
     * \par
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates four concurrent warp-wide exclusive prefix max scans within a block of
     * 128 threads (one per each of the 32-thread warps).
     * \par
     * \code
     * #include <cub/cub.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize WarpScan for type int
     *     typedef cub::WarpScan<int> WarpScan;
     *
     *     // Allocate WarpScan shared memory for 4 warps
     *     __shared__ typename WarpScan::TempStorage temp_storage[4];
     *
     *     // Obtain one input item per thread
     *     int thread_data = ...
     *
     *     // Compute exclusive warp-wide prefix max scans
     *     int warp_id = threadIdx.x / 32;
     *     WarpScan(temp_storage[warp_id]).ExclusiveScan(thread_data, thread_data, INT_MIN, cub::Max());
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{0, -1, 2, -3, ..., 126, -127}</tt>.
     * The corresponding output \p thread_data in the first warp would be
     * <tt>INT_MIN, 0, 0, 2, ..., 28, 30</tt>, the output for the second warp would be <tt>30, 32, 32, 34, ..., 60, 62</tt>, etc.
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
            T input,              ///< [in] Calling thread's input item.
            T & exclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
            T initial_value,      ///< [in] Initial value to seed the exclusive scan
            ScanOp scan_op)            ///< [in] Binary scan operator
    {
        InternalWarpScan internal(temp_storage);

        T inclusive_output;
        internal.InclusiveScan(input, inclusive_output, scan_op);

        internal.Update(
                input,
                inclusive_output,
                exclusive_output,
                scan_op,
                initial_value,
                cub::Int2Type<IS_INTEGER>());
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor across the calling warp.  Because no initial value is supplied, the \p output computed for <em>warp-lane</em><sub>0</sub> is undefined.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * \par
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates four concurrent warp-wide exclusive prefix max scans within a block of
     * 128 threads (one per each of the 32-thread warps).
     * \par
     * \code
     * #include <cub/cub.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize WarpScan for type int
     *     typedef cub::WarpScan<int> WarpScan;
     *
     *     // Allocate WarpScan shared memory for 4 warps
     *     __shared__ typename WarpScan::TempStorage temp_storage[4];
     *
     *     // Obtain one input item per thread
     *     int thread_data = ...
     *
     *     // Compute exclusive warp-wide prefix max scans
     *     int warp_aggregate;
     *     int warp_id = threadIdx.x / 32;
     *     WarpScan(temp_storage[warp_id]).ExclusiveScan(thread_data, thread_data, cub::Max(), warp_aggregate);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{0, -1, 2, -3, ..., 126, -127}</tt>.
     * The corresponding output \p thread_data in the first warp would be
     * <tt>?, 0, 0, 2, ..., 28, 30</tt>, the output for the second warp would be <tt>?, 32, 32, 34, ..., 60, 62</tt>, etc.
     * (The output \p thread_data in warp lane<sub>0</sub> is undefined.)  Furthermore, \p warp_aggregate would be assigned \p 30 for threads in the first warp, \p 62 for threads
     * in the second warp, etc.
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
            T input,              ///< [in] Calling thread's input item.
            T & exclusive_output,   ///< [out] Calling thread's output item.  May be aliased with \p input.
            ScanOp scan_op,            ///< [in] Binary scan operator
            T & warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        InternalWarpScan internal(temp_storage);

        T inclusive_output;
        internal.InclusiveScan(input, inclusive_output, scan_op);

        internal.Update(
                input,
                inclusive_output,
                exclusive_output,
                warp_aggregate,
                scan_op,
                cub::Int2Type<IS_INTEGER>());
    }


    /**
     * \brief Computes an exclusive prefix scan using the specified binary scan functor across the calling warp.  Also provides every thread with the warp-wide \p warp_aggregate of all inputs.
     *
     * \par
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates four concurrent warp-wide exclusive prefix max scans within a block of
     * 128 threads (one per each of the 32-thread warps).
     * \par
     * \code
     * #include <cub/cub.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize WarpScan for type int
     *     typedef cub::WarpScan<int> WarpScan;
     *
     *     // Allocate WarpScan shared memory for 4 warps
     *     __shared__ typename WarpScan::TempStorage temp_storage[4];
     *
     *     // Obtain one input item per thread
     *     int thread_data = ...
     *
     *     // Compute exclusive warp-wide prefix max scans
     *     int warp_aggregate;
     *     int warp_id = threadIdx.x / 32;
     *     WarpScan(temp_storage[warp_id]).ExclusiveScan(thread_data, thread_data, INT_MIN, cub::Max(), warp_aggregate);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{0, -1, 2, -3, ..., 126, -127}</tt>.
     * The corresponding output \p thread_data in the first warp would be
     * <tt>INT_MIN, 0, 0, 2, ..., 28, 30</tt>, the output for the second warp would be <tt>30, 32, 32, 34, ..., 60, 62</tt>, etc.
     * Furthermore, \p warp_aggregate would be assigned \p 30 for threads in the first warp, \p 62 for threads
     * in the second warp, etc.
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(
            T input,              ///< [in] Calling thread's input item.
            T & exclusive_output,  ///< [out] Calling thread's output item.  May be aliased with \p input.
            T initial_value,      ///< [in] Initial value to seed the exclusive scan
            ScanOp scan_op,            ///< [in] Binary scan operator
            T & warp_aggregate)    ///< [out] Warp-wide aggregate reduction of input items.
    {
        InternalWarpScan internal(temp_storage);

        T inclusive_output;
        internal.InclusiveScan(input, inclusive_output, scan_op);

        internal.Update(
                input,
                inclusive_output,
                exclusive_output,
                warp_aggregate,
                scan_op,
                initial_value,
                cub::Int2Type<IS_INTEGER>());
    }


    //@}  end member group
    #endif  // false


    /******************************************************************//**
     * \name Combination (inclusive & exclusive) prefix scans
     *********************************************************************/
    //@{


    /**
     * \brief Computes both inclusive and exclusive prefix scans using the specified binary scan functor across the calling warp.  Because no initial value is supplied, the \p exclusive_output computed for <em>warp-lane</em><sub>0</sub> is undefined.
     *
     * \par
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates four concurrent warp-wide exclusive prefix max scans within a block of
     * 128 threads (one per each of the 32-thread warps).
     * \par
     * \code
     * #include <cub/cub.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize WarpScan for type int
     *     typedef cub::WarpScan<int> WarpScan;
     *
     *     // Allocate WarpScan shared memory for 4 warps
     *     __shared__ typename WarpScan::TempStorage temp_storage[4];
     *
     *     // Obtain one input item per thread
     *     int thread_data = ...
     *
     *     // Compute exclusive warp-wide prefix max scans
     *     int inclusive_partial, exclusive_partial;
     *     WarpScan(temp_storage[warp_id]).Scan(thread_data, inclusive_partial, exclusive_partial, cub::Max());
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{0, -1, 2, -3, ..., 126, -127}</tt>.
     * The corresponding output \p inclusive_partial in the first warp would be
     * <tt>0, 0, 2, 2, ..., 30, 30</tt>, the output for the second warp would be <tt>32, 32, 34, 34, ..., 62, 62</tt>, etc.
     * The corresponding output \p exclusive_partial in the first warp would be
     * <tt>?, 0, 0, 2, ..., 28, 30</tt>, the output for the second warp would be <tt>?, 32, 32, 34, ..., 60, 62</tt>, etc.
     * (The output \p thread_data in warp lane<sub>0</sub> is undefined.)
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void Scan(
            T input,              ///< [in] Calling thread's input item.
            T & inclusive_output,  ///< [out] Calling thread's inclusive-scan output item.
            T & exclusive_output,  ///< [out] Calling thread's exclusive-scan output item.
            ScanOp scan_op,             ///< [in] Binary scan operator
            ScanDir scanDir)
    {
        InternalWarpScan internal(temp_storage);

        internal.InclusiveScan(input, inclusive_output, scan_op, scanDir);

        internal.Update(input,
                        inclusive_output,
                        exclusive_output,
                        scan_op,
                        scanDir);
    }


    #if false
    /**
     * \brief Computes both inclusive and exclusive prefix scans using the specified binary scan functor across the calling warp.
     *
     * \par
     *  - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates four concurrent warp-wide prefix max scans within a block of
     * 128 threads (one per each of the 32-thread warps).
     * \par
     * \code
     * #include <cub/cub.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize WarpScan for type int
     *     typedef cub::WarpScan<int> WarpScan;
     *
     *     // Allocate WarpScan shared memory for 4 warps
     *     __shared__ typename WarpScan::TempStorage temp_storage[4];
     *
     *     // Obtain one input item per thread
     *     int thread_data = ...
     *
     *     // Compute inclusive warp-wide prefix max scans
     *     int warp_id = threadIdx.x / 32;
     *     int inclusive_partial, exclusive_partial;
     *     WarpScan(temp_storage[warp_id]).Scan(thread_data, inclusive_partial, exclusive_partial, INT_MIN, cub::Max());
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{0, -1, 2, -3, ..., 126, -127}</tt>.
     * The corresponding output \p inclusive_partial in the first warp would be
     * <tt>0, 0, 2, 2, ..., 30, 30</tt>, the output for the second warp would be <tt>32, 32, 34, 34, ..., 62, 62</tt>, etc.
     * The corresponding output \p exclusive_partial in the first warp would be
     * <tt>INT_MIN, 0, 0, 2, ..., 28, 30</tt>, the output for the second warp would be <tt>30, 32, 32, 34, ..., 60, 62</tt>, etc.
     *
     * \tparam ScanOp     <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    __device__ __forceinline__ void Scan(
            T input,              ///< [in] Calling thread's input item.
            T & inclusive_output,  ///< [out] Calling thread's inclusive-scan output item.
            T & exclusive_output,  ///< [out] Calling thread's exclusive-scan output item.
            T initial_value,      ///< [in] Initial value to seed the exclusive scan
            ScanOp scan_op,            ///< [in] Binary scan operator
            ScanDir scanDir)
    {
        InternalWarpScan internal(temp_storage);

        internal.InclusiveScan(input, inclusive_output, scan_op, scanDir);

        internal.Update(
                input,
                inclusive_output,
                exclusive_output,
                scan_op,
                initial_value,
                scanDir);
    }
    #endif  // false



    //@}  end member group
    /******************************************************************//**
     * \name Data exchange
     *********************************************************************/
    //@{

    /**
     * \brief Broadcast the value \p input from <em>warp-lane</em><sub><tt>src_lane</tt></sub> to all lanes in the warp
     *
     * \par
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates the warp-wide broadcasts of values from
     * lanes<sub>0</sub> in each of four warps to all other threads in those warps.
     * \par
     * \code
     * #include <cub/cub.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize WarpScan for type int
     *     typedef cub::WarpScan<int> WarpScan;
     *
     *     // Allocate WarpScan shared memory for 4 warps
     *     __shared__ typename WarpScan::TempStorage temp_storage[4];
     *
     *     // Obtain one input item per thread
     *     int thread_data = ...
     *
     *     // Broadcast from lane0 in each warp to all other threads in the warp
     *     int warp_id = threadIdx.x / 32;
     *     thread_data = WarpScan(temp_storage[warp_id]).Broadcast(thread_data, 0);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{0, 1, 2, 3, ..., 127}</tt>.
     * The corresponding output \p thread_data will be
     * <tt>{0, 0, ..., 0}</tt> in warp<sub>0</sub>,
     * <tt>{32, 32, ..., 32}</tt> in warp<sub>1</sub>,
     * <tt>{64, 64, ..., 64}</tt> in warp<sub>2</sub>, etc.
     */
    __device__ __forceinline__ T Broadcast(
            T input,              ///< [in] The value to broadcast
            unsigned int src_lane)           ///< [in] Which warp lane is to do the broadcasting
    {
        return InternalWarpScan(temp_storage).Broadcast(input, src_lane);
    }

    //@}  end member group

};

}  // namespace ndmamba
