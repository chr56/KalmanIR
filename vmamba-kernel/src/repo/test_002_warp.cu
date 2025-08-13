#include <cub/block/block_scan.cuh>
#include <cub/util_ptx.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "utils/cuda_utils.h"
#include "scan/block_scan.cuh"


template <typename T>
__global__ void test()
{
    using Scan = mamband::SegWarpScan<T, 8>;
    typename Scan::TempStorage tempStorage;
    Scan scan(tempStorage);

    T input = cub::LaneId() / 8;
    T inclusiveOutput;
    T segmentAggregate;
    scan.InclusiveScan(input, inclusiveOutput, cub::Sum(), segmentAggregate);
    printf("lane %u inclusiveOutput = %f warpAggregate = %f\n", cub::LaneId(), inclusiveOutput, segmentAggregate);
}


int main()
{
//    thrust::device_vector<float> d_out(8 * 16, 1.0f);
    test<float><<<1, 32>>>();
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
