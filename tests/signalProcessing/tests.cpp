#include <gtest/gtest.h>
#include "readFile.hh"
#include "upsample.hh"
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "kernals.cuh"


TEST(SIGNAL_PROCESSING, UPSAMPLE)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    printf("Read file\n");
    std::vector<float> h_data = readFile("./vectors/inputSignal2.bin");
    EXPECT_NE(h_data.size(), 0u);
    printf("h_data.size: %lu\n", h_data.size());


    size_t upsampleFactor  = 2;
    float* d_Input = allocateDeviceMemory(h_data.size(), upsampleFactor);
    EXPECT_NE(d_Input, nullptr);
    printf("d_Input: %p\n", d_Input);
    err = cudaMemcpy(d_Input, h_data.data(), h_data.size()*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector from host to device (error code %s)!\n",
            cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    float* d_Output = allocateDeviceMemory(h_data.size(), upsampleFactor);
    EXPECT_NE(d_Output, nullptr);
    printf("d_Output: %p\n", d_Output);

    // int* d_UpSampleFactor = NULL;
    // err = cudaMalloc((void **)&d_UpSampleFactor, sizeof(int));
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
    //     cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // err = cudaMemcpy(d_UpSampleFactor, &upsampleFactor, sizeof(int), cudaMemcpyHostToDevice);
    // if (err != cudaSuccess) {
    // fprintf(stderr,
    //         "Failed to copy vector from host to device (error code %s)!\n",
    //         cudaGetErrorString(err));
    //   exit(EXIT_FAILURE);
    // }




    // Launch the Vector Add CUDA Kernel
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (h_data.size() + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    // //void deviceUpsample(const float *output, const float *input, int numElements, int upsampleFactor)
    // int numElements = h_data.size();
    // deviceUpsample<<<blocksPerGrid, threadsPerBlock>>>(d_Output, d_Input, numElements, upsampleFactor);
    int numElements = h_data.size();
    hostUpsample(d_Output, d_Input, numElements, upsampleFactor);


    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    std::vector<float> h_results(h_data.size());

    err = cudaMemcpy(h_results.data(), d_Input, h_data.size()*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    ASSERT_EQ(h_data.size(), h_results.size());

    for(size_t i = 0; i < h_data.size(); i++)
    {
        EXPECT_FLOAT_EQ(h_data[i], h_results[i]);
    }





    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    std::vector<float> h_upsample(h_data.size()*upsampleFactor);

    err = cudaMemcpy(h_upsample.data(), d_Output, h_upsample.size()*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy d_Output vector from device to host (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    ASSERT_EQ(h_data.size()*upsampleFactor, h_upsample.size());

    for(size_t i = 0; i < h_upsample.size(); i++)
    {
        //EXPECT_FLOAT_EQ(h_data[i], h_results[i]);
        printf("[%u]: %f\n", i, h_upsample[i]);
    }






    cleanupDeviceMemory(d_Input);
}