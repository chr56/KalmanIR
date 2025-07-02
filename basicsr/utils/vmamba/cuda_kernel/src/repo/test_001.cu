#include <cub/block/block_scan.cuh>
#include <cub/util_ptx.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"
#include "scan/scan.cuh"


#if false
template <typename T>
__global__ void testCubWarpScanShfl()
{
    // Specialize WarpScan for type int
    typedef cub::WarpScan<T> WarpScan;

    // Allocate WarpScan shared memory for 4 warps
    __shared__ typename WarpScan::TempStorage temp_storage[4];

    // Obtain one input item per thread
    T thread_data = 1.0f;
    T inital_val = 0.0f;

    // Compute inclusive warp-wide prefix max scans
    int warp_id = threadIdx. x / 32;

    T inclusive_partial;
    T exclusive_partial;

    WarpScan(temp_storage[warp_id]).Scan(
        thread_data,
        inclusive_partial,
        exclusive_partial,
        inital_val,
        cub::Sum());

    printf("thread_data[%d] = %f %f\n", threadIdx.x, inclusive_partial, exclusive_partial);
}
#endif  // false

#if false
template <typename T>
__global__ void testMambaWarpReverseScan()
{
    // Specialize WarpScan for type int
    using WarpScan = WarpReverseScan<T, 32>;

    // Obtain one input item per thread
    T thread_data = static_cast<T>(threadIdx.x);

    // Compute inclusive warp-wide prefix max scans
    int warp_id = threadIdx. x / 32;

    T inclusive_partial;
    T exclusive_partial;

    WarpScan().ReverseScan(
        thread_data,
        inclusive_partial,
        exclusive_partial,
        cub::Sum()
    );

    printf("thread_data[%d] = %f\n", threadIdx.x, inclusive_partial);
}
#endif  // false


// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct SSMScanPrefixCallbackOp {
    using scan_t = float;
    scan_t running_prefix;
    // Constructor
    __device__ SSMScanPrefixCallbackOp(scan_t running_prefix_) : running_prefix(running_prefix_) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ scan_t operator()(scan_t block_aggregate) {
        scan_t old_prefix = running_prefix;
        running_prefix = running_prefix + block_aggregate;
        return old_prefix;
    }
};


//template <typename T>
//__global__ void testMambaInclusiveReverseScan()
//{
//    T thread_data[1];
////    thread_data[0] = threadIdx.x;
//thread_data[0] = 1.0f;
//    T thread_output[1];
//
//    using BlockReverseScan = BlockReverseScan<T, 64>;
//
//    __shared__ typename BlockReverseScan::TempStorage temp_storage;
//    auto callback_op = SSMScanPrefixCallbackOp(0.0f);
//
//    BlockReverseScan(temp_storage).InclusiveReverseScan(thread_data, thread_output, cub::Sum(), callback_op);
//
//    printf("thread_output[%d] = %f\n", threadIdx.x, thread_output[0]);
//}

#if false
template <typename T>
__global__ void twoDimScanTest()
{
    constexpr int kSteps = 5;
    constexpr int kWarpSize = 32;
    constexpr unsigned kMemberMask = 0xffffffffu;

    int tid = threadIdx.x;
    int laneIdx = tid % kWarpSize;
    int warpIdx = tid / kWarpSize;

    auto scanOp = cub::Sum();

    T input = 1.0f;

    #pragma unroll
    for (int step = 0; step < kSteps; ++step)
    {
        int firstLane = step < 3 ? ((tid >> 3) << 3) : 0;
        int offset = 1 << step;

        T temp = cub::ShuffleUp<kWarpSize>(
            input,
            offset,
            firstLane,
            kMemberMask
        );

        // Perform scan op if from a valid peer
        T output = scanOp(temp, input);

        if (static_cast<int>(laneIdx) < firstLane + offset)
        {
            output = input;
        }

        input = output;
    }

    printf("thread %d output = %f\n", tid, input);
}
#endif  // false


template <typename T>
__global__ void test(T * __restrict__ out)
{
    constexpr int kSteps = 5;
    constexpr int kWarpSize = 32;
    constexpr unsigned kMemberMask = 0xffffffffu;

    int tid = threadIdx.x;
    int laneIdx = tid % kWarpSize;
    int warpIdx = tid / kWarpSize;

    auto scanOp = cub::Sum();

    constexpr int kInputDim = 2;
    T input[kInputDim][kInputDim];

    // fake load & initial scan.
    for (int i = 0; i < kInputDim; ++i)
    {
        for (int j = 0; j < kInputDim; ++j)
        {
            input[i][j] = 1.0f;
        }
    }

    for (int i = 0; i < kInputDim; ++i)
    {
        for (int j = 0; j < kInputDim; ++j)
        {
            if (0 < j)
            {
                input[i][j] += input[i][j - 1];
            }
        }

        for (int j = 0; j < kInputDim; ++j)
        {
            if (0 < i)
            {
                input[i][j] += input[i - 1][j];
            }
        }
    }

    __syncwarp();

    #pragma unroll
    for (int step = 0; step < kSteps; ++step)
    {
        int firstLane = step < 3 ? ((tid >> 3) << 3) : 0;
        int offset = 1 << step;

        T temp = cub::ShuffleUp<kWarpSize>(
            input,
            offset,
            firstLane,
            kMemberMask
        );

        // Perform scan op if from a valid peer
        T output = scanOp(temp, input);

        if (static_cast<int>(laneIdx) < firstLane + offset)
        {
            output = input;
        }

        input = output;
    }
}


int main()
{
    thrust::device_vector<float> d_out(8 * 16);
    test<float><<<1, 32>>>(thrust::raw_pointer_cast(d_out.data()));
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::host_vector<float> h_out = d_out;

    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            printf("%f ", h_out[i * 16 + j]);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}
