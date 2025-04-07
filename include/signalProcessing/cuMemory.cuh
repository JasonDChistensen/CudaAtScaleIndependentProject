
#ifndef CU_MEMORY_H
#define CU_MEMORY_H

#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if (error != cudaSuccess)                                                   \
    {                                                                           \
        printf("CUDA Error: %s:%d, ", __FILE__, __LINE__);                      \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));      \
        exit(1);                                                                \
    }                                                                           \
}       

#define CHECK_NPP(call)                                                         \
{                                                                               \
    const NppStatus error = call;                                               \
    if (error != NPP_SUCCESS)                                                   \
    {                                                                           \
        printf("NPP Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code:%d\n", error);                                             \
        exit(1);                                                                \
    }                                                                           \
}

#define CHECK_CUBLAS(call)                                                      \
{                                                                               \
    const cublasStatus_t error = call;                                          \
    if (error != CUBLAS_STATUS_SUCCESS)                                         \
    {                                                                           \
        printf("cuBLAS Error: %s:%d, ", __FILE__, __LINE__);                    \
        printf("code:%d\n", error);                                             \
        exit(1);                                                                \
    }                                                                           \
}

 
template <typename T>
class cuMemory
{
public:
    cuMemory(size_t nLength, std::vector<T> init):cuMemory(nLength)
    {
        cudaError_t err = cudaMemcpy(m_data, init.data(), init.size()*sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr,
            "Failed to copy d_Output vector from device to host (error code %s)!\n",
            cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
    }
    cuMemory(size_t nLength): m_length(nLength)
    {
        //printf("cuMemory, constructor, nLength: %lu\n", nLength);
        cudaError_t err = cudaMalloc((void **)&m_data, nLength*sizeof(T));
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate device memory (error code %s)!\n",
            cudaGetErrorString(err));
            m_data = nullptr;
            //exit(EXIT_FAILURE);
        }
    }
    ~cuMemory()
    {
        //printf("cuMemory, desstructor\n");
        // Free device global memory
        cudaError_t err = cudaFree(m_data);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to free device vector (error code %s)!\n",
            cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
    }
    T *get_ptr(void){ return m_data; }
    size_t get_length(void){ return m_length;}
    std::vector<T> get_data(void)
    {
        std::vector<T> result(m_length);
        cudaError_t err = cudaMemcpy(result.data(), m_data, m_length*sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr,
            "Failed to copy d_Output vector from device to host (error code %s)!\n",
            cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        return result;
    }

private:
    T *m_data;
    size_t m_length;
};

#endif // CU_MEMORY_H