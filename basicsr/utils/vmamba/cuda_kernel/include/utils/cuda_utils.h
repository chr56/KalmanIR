#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H


// AMGX API error checking
#define AMGX_CHECK(err)                                                                            \
    do {                                                                                           \
        AMGX_RC err_ = (err);                                                                      \
        if (err_ != AMGX_RC_OK) {                                                                  \
            char checkBuf[1024] = {'\0'};                                                          \
            std::sprintf(checkBuf, "AMGX error %d at %s:%d\n", err_, __FILE__, __LINE__);          \
            cudaDeviceReset();                                                                     \
            throw std::runtime_error(checkBuf);                                                    \
        }                                                                                          \
    } while (false)


// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            char checkBuf[1024] = {'\0'};                                                          \
            std::sprintf(checkBuf, "cuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);        \
            cudaDeviceReset();                                                                     \
            throw std::runtime_error(checkBuf);                                                    \
        }                                                                                          \
    } while (false)


// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            char checkBuf[1024] = {'\0'};                                                          \
            std::sprintf(checkBuf, "%s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            cudaDeviceReset();                                                                     \
            throw std::runtime_error(checkBuf);                                                    \
        }                                                                                          \
    } while (false)


// CUDA kernel function launch status checking
#define CUDA_CHECK_LAST_ERROR()                                                                    \
    do {                                                                                           \
        cudaError_t err_ = cudaGetLastError();                                                     \
        if (err_ != cudaSuccess) {                                                                 \
            char checkBuf[1024] = {'\0'};                                                          \
            std::sprintf(checkBuf, "%s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            cudaDeviceReset();                                                                     \
            throw std::runtime_error(checkBuf);                                                    \
        }                                                                                          \
    } while (false)


// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            char checkBuf[1024] = {'\0'};                                                          \
            std::sprintf(checkBuf, "cuSOLVER error %d at %s:%d\n", err_, __FILE__, __LINE__);      \
            cudaDeviceReset();                                                                     \
            throw std::runtime_error(checkBuf);                                                    \
        }                                                                                          \
    } while (false)


// cublas API error checking
#define CUSPARSE_CHECK(err)                                                                        \
    do {                                                                                           \
        cusparseStatus_t err_ = (err);                                                             \
        if (err_ != CUSPARSE_STATUS_SUCCESS) {                                                     \
            char checkBuf[1024] = {'\0'};                                                          \
            std::sprintf(checkBuf, "cuSPARSE error %d at %s:%d\n", err_, __FILE__, __LINE__);      \
            cudaDeviceReset();                                                                     \
            throw std::runtime_error(checkBuf);                                                    \
        }                                                                                          \
    } while (false)


#endif  // CUDA_UTILS_H
