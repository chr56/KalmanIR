#include <numeric>
#include <random>

#include <cub/block/block_scan.cuh>
#include <cub/util_ptx.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"
#include "scan/block_scan.cuh"
#include "scan/commons.h"



template <typename T>
struct ScanOp
{
    __device__ __forceinline__ T operator()(const T & a, const T & b) = delete;
};


template <>
struct ScanOp<float>
{
    __device__ __forceinline__ float operator()(const float & a, const float & b)
    {
        return a + b;
    }
};


template <>
struct ScanOp<float2>
{
    __device__ __forceinline__ float2 operator()(const float2 & a, const float2 & b)
    {
        return {a.x + b.x, a.y + b.y};
    }
};


template <typename ScanOp, typename T>
struct BlockPrefixCallbackOp
{
    __device__ BlockPrefixCallbackOp(T runningPrefix) : runningPrefix(runningPrefix) {}

    __device__ T operator()(T blockAggregate)
    {
        T oldPrefix = runningPrefix;
        runningPrefix = ScanOp()(runningPrefix, blockAggregate);
        return oldPrefix;
    }

    T runningPrefix = 0;
};


template <typename T>
__global__ void scan()
{
    using Scan = cub::BlockScan<T, 64, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockPrefixCallbackOp = BlockPrefixCallbackOp<cub::Sum, T>;

    __shared__ typename Scan::TempStorage tempStorage;
    Scan scan(tempStorage);

    BlockPrefixCallbackOp blockPrefixCallbackOp(0);

    T input = 1;
    scan.InclusiveScan(input, input, cub::Sum(), blockPrefixCallbackOp);

    input = 1;
    scan.InclusiveScan(input, input, cub::Sum(), blockPrefixCallbackOp);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("tid %d input = %f\n", tid, input);

    if (tid == 0)
    {
        printf("blockPrefixCallbackOp.runningPrefix = %f\n", blockPrefixCallbackOp.runningPrefix);
    }
}


int main()
{
    scan<float><<<1, 64>>>();
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
