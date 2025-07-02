#include <iomanip>
#include <iostream>
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
    __host__ __device__ __forceinline__ T operator()(const T & a, const T & b) const = delete;
};


template <>
struct ScanOp<float>
{
    __host__ __device__ __forceinline__ float operator()(const float & a, const float & b) const
    {
        return a + b;
    }
};


template <>
struct ScanOp<int>
{
    __host__ __device__ __forceinline__ int operator()(const int & a, const int & b) const
    {
        return a + b;
    }
};


template <typename T, typename ScanOp>
struct BlockPrefixCallbackOp
{
    __host__ __device__ BlockPrefixCallbackOp()
    {

    }

    __host__ __device__ T operator()(T blockAggregate)
    {
        T oldPrefix = runningPrefix;
        runningPrefix = scanOp(runningPrefix, blockAggregate);
        return oldPrefix;
    }

    T runningPrefix = 0;
    ScanOp scanOp;
};


template <int kMatrixX, int kMatrixY, int kBlockDim, int kSegLen, int kThreadSpan = 2, typename T>
__global__ void scan(const T * __restrict__ src, int nx, int ny, T * __restrict__ dst)
{
    constexpr int kWarpThreads = 32;

    using Scan = ndmamba::SegBlockScan<T, kSegLen, kBlockDim, kBlockDim>;
    __shared__ typename Scan::TempStorage tempStorage;
    Scan scan(tempStorage);

    ScanOp<T> scanOp;
    using BlockPrefixCallbackOp = BlockPrefixCallbackOp<T, ScanOp<T>>;

    T thread_data[kThreadSpan][kThreadSpan];
    BlockPrefixCallbackOp prefix_ops_for_horizontal_scan[kThreadSpan];
    BlockPrefixCallbackOp prefix_ops_for_vertical_scan[kThreadSpan];

    constexpr int kChunkDim = kBlockDim * kThreadSpan;
    int chunksAcrossDimY = ny / kChunkDim;
    int chunksAcrossDimX = nx / kChunkDim;

    #pragma unroll
    for (int chunk_y_idx = chunksAcrossDimY - 1; 0 <= chunk_y_idx; --chunk_y_idx)
    {
        #pragma unroll
        for (int chunk_x_idx = chunksAcrossDimX - 1; 0 <= chunk_x_idx; --chunk_x_idx)
        {
            const int thread_x = kChunkDim * chunk_x_idx + threadIdx.x * kThreadSpan;
            const int thread_y = kChunkDim * chunk_y_idx + threadIdx.y * kThreadSpan;

            for (int yy = 0; yy < kThreadSpan; ++yy)
            {
                for (int xx = 0; xx < kThreadSpan; ++xx)
                {
                    const int gx = thread_x + xx;
                    const int gy = thread_y + yy;
                    thread_data[yy][xx] = src[gy * nx + gx];
                }
            }

            scan.InclusiveScan(thread_data, thread_data, scanOp, prefix_ops_for_vertical_scan, ndmamba::kVerticalReversed);
//            scan.InclusiveScan(input, input, scanOp, xCallback, ndmamba::kHorizontal);

            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                printf("yi == %d xi == %d\n", chunk_y_idx, chunk_x_idx);
            }

            if (chunk_y_idx == 0 && chunk_x_idx == 0)
            {
                printf("chunk y=%d x=%d tid y=%d x=%d x callback op %f %f\n",
                       chunk_y_idx, chunk_x_idx, threadIdx.y, threadIdx.x,
                       prefix_ops_for_vertical_scan[0].runningPrefix,
                       prefix_ops_for_vertical_scan[1].runningPrefix);
            }

            for (int yy = 0; yy < kThreadSpan; ++yy)
            {
                for (int xx = 0; xx < kThreadSpan; ++xx)
                {
                    const int gx = thread_x + xx;
                    const int gy = thread_y + yy;
                    dst[gy * nx + gx] = thread_data[yy][xx];
//                    dst[gy * nx + gx] = threadIdx.y + static_cast<float>(threadIdx.x) / 1000.0f;
                }
            }
        }
    }
}


int main()
{
    // Each chunk is of size (kBlockDim x kThreadSpan, kBlockDim x kThreadSpan)
    constexpr int kBlockDim = 8;
    constexpr dim3 kBlock(kBlockDim, kBlockDim);
    constexpr int kThreadSpan = 2;
    constexpr int kSegLen = 8;

    // Input data is of size (kBlockDim x kThreadSpan x xChunks, kBlockDim x kThreadSpan x yChunks)
    constexpr int xChunks = 1;
    constexpr int yChunks = 2;
    constexpr dim3 kMatrixSize(kBlockDim * kThreadSpan * xChunks, kBlockDim * kThreadSpan * yChunks);
    constexpr bool kRandInput = false;

    using T = float;
    std::vector<T> matBuf(kMatrixSize.x * kMatrixSize.y, 1);

    if constexpr (kRandInput)
    {
        auto seed = std::random_device()();
        auto e = std::default_random_engine(seed);
        auto d = std::normal_distribution<float>(0.0f, 1.0f);
//        auto d = std::uniform_int_distribution(1, 100000);
        auto g = [&d, &e]()
        {
            return d(e);
        };
        std::generate(matBuf.begin(), matBuf.end(), g);
    }

    thrust::host_vector<T> hostScanSrc = matBuf;

    thrust::device_vector<T> devScanSrc = hostScanSrc;
    thrust::device_vector<T> devScanDst(kMatrixSize.x * kMatrixSize.y, 0.0f);
    thrust::device_vector<T> devHoriAgg(kMatrixSize.x * kMatrixSize.y, 0.0f);
    thrust::device_vector<T> devVertAgg(kMatrixSize.x * kMatrixSize.y, 0.0f);

    scan<kMatrixSize.x, kMatrixSize.y, kBlockDim, kSegLen, kThreadSpan><<<1, kBlock>>>(
            thrust::raw_pointer_cast(devScanSrc.data()),
            kMatrixSize.x,
            kMatrixSize.y,
            thrust::raw_pointer_cast(devScanDst.data())
    );
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::host_vector<T> hostScanDst = devScanDst;

    auto mat = [kMatrixSize, &matBuf](int i, int j) mutable -> T &
    {
        return matBuf[i * kMatrixSize.x + j];
    };

    auto hRes = [kMatrixSize, &hostScanDst](int i, int j) mutable -> T &
    {
        return hostScanDst[i * kMatrixSize.x + j];
    };

//    for (int j = 1; j < kMatrixSize.x; ++j)
//    {
//        mat(0, j) += mat(0, j - 1);
//    }
//
//    for (int i = 1; i < kMatrixSize.y; ++i)
//    {
//        mat(i, 0) += mat(i - 1, 0);
//    }
//
//    for (int i = 1; i < kMatrixSize.y; ++i)
//    {
//        for (int j = 1; j < kMatrixSize.x; ++j)
//        {
//            mat(i, j) += mat(i, j - 1) + mat(i - 1, j) - mat(i - 1, j - 1);
//        }
//    }

//    bool isCorrect = true;
//    for (int i = 0; i < kMatrixSize.y; ++i)
//    {
//        for (int j = 0; j < kMatrixSize.x; ++j)
//        {
//            if (1e-4f < std::abs(mat(i, j) - hRes(i, j)))
//            {
//                isCorrect = false;
//            }
//        }
//    }
//    printf("%s\n", isCorrect ? "Correct" : "WRONG!!!");
//
//    printf("cpu\n");
//    for (int i = 0; i < kMatrixSize.y; ++i)
//    {
//        for (int j = 0; j < kMatrixSize.x; ++j)
//        {
//            printf("%13.6f ", mat(i, j));
//        }
//        printf("\n");
//    }
//    printf("\n");

    printf("scan\n");
    for (int i = 0; i < kMatrixSize.y; ++i)
    {
        for (int j = 0; j < kMatrixSize.x; ++j)
        {
            printf("%13.6f ", hRes(i, j));
        }
        printf("\n");
    }
    printf("\n");

//    void * d_temp_storage = NULL;
//    thrust::device_vector<T> devScanDst2(kMatrixSize.x * kMatrixSize.y, 0.0f);
//    T * d_in = thrust::raw_pointer_cast(devScanSrc.data());
//    T * d_out = thrust::raw_pointer_cast(devScanDst2.data());
//    size_t temp_storage_bytes = 0;
//    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes,
//                                   d_in, d_out,
//                                   ScanOp<T>(), kMatrixSize.x * kMatrixSize.y);
//    cudaMalloc(&d_temp_storage, temp_storage_bytes);
//    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes,
//                                   d_in, d_out,
//                                   ScanOp<T>(), kMatrixSize.x * kMatrixSize.y);
//    T res;
//    CUDA_CHECK(cudaMemcpy(&res, d_out + kMatrixSize.x * kMatrixSize.y - 1, sizeof(T), cudaMemcpyDeviceToHost));
//    printf("cub device scan ");
//    std::cout << std::setprecision(10) << res << '\n';
//
//    T tres2 = thrust::reduce(devScanSrc.cbegin(), devScanSrc.cend(), static_cast<T>(0));
//    printf("thrust dev reduce ");
//    std::cout << std::setprecision(10) << tres2 << '\n';
//
//    T cpures = std::accumulate(hostScanSrc.cbegin(), hostScanSrc.cend(), static_cast<T>(0));
//    printf("stl accumulate ");
//    std::cout << std::setprecision(10) << cpures << '\n';
//
//    T tres = thrust::reduce(hostScanSrc.cbegin(), hostScanSrc.cend(), static_cast<T>(0));
//    printf("thrust host reduce ");
//    std::cout << std::setprecision(10) << tres << '\n';
//
//    printf("cpu scan ");
//    std::cout << mat(kMatrixSize.y - 1, kMatrixSize.x - 1) << '\n';
//
//    printf("seg pscan ");
//    std::cout << hRes(0, 0) << '\n';

    return EXIT_SUCCESS;
}
