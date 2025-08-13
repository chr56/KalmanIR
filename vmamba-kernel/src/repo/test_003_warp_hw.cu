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
struct ScanOp<float2>
{
    __device__ __forceinline__ float2 operator()(const float2 & a, const float2 & b)
    {
        return {a.x + b.x, a.y + b.y};
    }
};


__global__ void test()
{
    using Scan = mamband::SegWarpScan<float2, 8>;
    typename Scan::TempStorage tempStorage;
    Scan scan(tempStorage);

    ScanOp<float2> scanOp;

    float2 input;
    input.x = cub::LaneId() / 8;
    input.y = cub::LaneId() % 8;

    float2 segAgg;

    scan.InclusiveScan(input, input, scanOp, segAgg, mamband::kHorizontal);
    printf("lane %u inclusiveOutput = %f %f warpAggregate = %f %f\n",
           cub::LaneId(),
           input.x, input.y,
           segAgg.x, segAgg.y);

    printf("\n");

    scan.InclusiveScan(input, input, scanOp, segAgg, mamband::kVertical);
    printf("lane %u inclusiveOutput = %f %f warpAggregate = %f %f\n",
           cub::LaneId(),
           input.x, input.y,
           segAgg.x, segAgg.y);
}


int main()
{
//    thrust::device_vector<float> d_out(8 * 16, 1.0f);
    test<<<1, 32>>>();
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
