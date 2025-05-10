
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
    cuMemory(std::vector<T> init):cuMemory(init.size())
    {
        CHECK(cudaMemcpy(m_data, init.data(), init.size()*sizeof(T), cudaMemcpyHostToDevice));
    }
    cuMemory(size_t length): m_length(length)
    {
        //printf("cuMemory, constructor, length: %zu, sizeof(T):%lu,  length*sizeof(T):%lu\n", 
        //m_length, sizeof(T), m_length*sizeof(T));
        CHECK(cudaMalloc((void **)&m_data, m_length*sizeof(T)));
    }
    ~cuMemory()
    {
        //printf("cuMemory, desstructor, m_data:%p\n", m_data);
        // Free device global memory
        CHECK(cudaFree(m_data));
    }
    T *data(void){ return m_data; }
    size_t size(void){ return m_length;}
    std::vector<T> get_data(void)
    {
        std::vector<T> result(m_length);
        CHECK(cudaMemcpy(result.data(), m_data, m_length*sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }

private:
    T *m_data;
    size_t m_length;
};

#endif // CU_MEMORY_H