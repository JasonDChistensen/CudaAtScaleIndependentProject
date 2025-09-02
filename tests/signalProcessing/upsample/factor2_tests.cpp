#include <gtest/gtest.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "readFile.hh"
#include "cuMemory.cuh"
#include "upsample.cuh"

TEST(FACTOR2, UPSAMPLE)
{
    std::vector<float> h_Input = readFile("./vectors/inputSignal2.bin");
    ASSERT_NE(h_Input.size(), 0u);
    printf("h_Input.size: %lu\n", h_Input.size());

    const size_t upsampleFactor  = 2;
    cuMemory<float> d_Input(h_Input.size() * upsampleFactor);

    CHECK(cudaMemcpy(d_Input.data(), h_Input.data(), h_Input.size()*sizeof(float), cudaMemcpyHostToDevice));

    cuMemory<float> d_Output(h_Input.size() * upsampleFactor);


    int numElements = h_Input.size();
    upSample::execute(d_Output.data(), d_Input.data(), numElements, upsampleFactor);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    std::vector<float> h_Expected_Output = readFile("./vectors/inputSignalUpsampled2.bin");
    ASSERT_NE(h_Expected_Output.size(), 0u);
    printf("h_Expected_Output.size: %lu\n", h_Expected_Output.size());


    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    std::vector<float> h_Output(h_Input.size()*upsampleFactor);

    CHECK(cudaMemcpy(h_Output.data(), d_Output.data(), h_Output.size()*sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_EQ(h_Input.size()*upsampleFactor, h_Output.size());
    ASSERT_EQ(h_Expected_Output.size(), h_Output.size());

    
    for(size_t i = 0; i < h_Output.size(); i++)
    {
        // printf("h_Output[%lu]: %f, h_Expected_Output[%lu]: %f\n",
        //     i, h_Output[i],i, h_Expected_Output[i]);
        ASSERT_FLOAT_EQ(h_Output[i], h_Expected_Output[i]);      
    }
}
