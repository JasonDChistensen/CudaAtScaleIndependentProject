#include <gtest/gtest.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "readFile.hh"
#include "upsample.cuh"
#include "interpolate.cuh"


TEST(SIGNAL_PROCESSING, UPSAMPLE)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    printf("Read file\n");
    std::vector<float> h_data = readFile("./vectors/inputSignal2.bin");
    ASSERT_NE(h_data.size(), 0u);
    printf("h_data.size: %lu\n", h_data.size());


    size_t upsampleFactor  = 2;
    float* d_Input = upSample::allocateDeviceMemory(h_data.size(), upsampleFactor);
    ASSERT_NE(d_Input, nullptr);
    printf("d_Input: %p\n", d_Input);
    err = cudaMemcpy(d_Input, h_data.data(), h_data.size()*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector from host to device (error code %s)!\n",
            cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    float* d_Output = upSample::allocateDeviceMemory(h_data.size(), upsampleFactor);
    ASSERT_NE(d_Output, nullptr);
    printf("d_Output: %p\n", d_Output);



    int numElements = h_data.size();
    upSample::execute(d_Output, d_Input, numElements, upsampleFactor);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // // Copy the device result vector in device memory to the host result vector
    // // in host memory.
    // printf("Copy output data from the CUDA device to the host memory\n");
    // std::vector<float> h_results(h_data.size());

    // err = cudaMemcpy(h_results.data(), d_Output, h_data.size()*sizeof(float), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) {
    //     fprintf(stderr,
    //         "Failed to copy vector C from device to host (error code %s)!\n",
    //         cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // ASSERT_EQ(h_data.size(), h_results.size());

    // for(size_t i = 0; i < h_data.size(); i++)
    // {
    //     EXPECT_FLOAT_EQ(h_data[i], h_results[i]);
    // }


    printf("Read file\n");
    std::vector<float> h_upsampleExpected = readFile("./vectors/inputSignalUpsampled2.bin");
    ASSERT_NE(h_upsampleExpected.size(), 0u);
    printf("h_upsampleExpected.size: %lu\n", h_upsampleExpected.size());





    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    std::vector<float> h_upsampleResults(h_data.size()*upsampleFactor);

    err = cudaMemcpy(h_upsampleResults.data(), d_Output, h_upsampleResults.size()*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy d_Output vector from device to host (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    ASSERT_EQ(h_data.size()*upsampleFactor, h_upsampleResults.size());
    ASSERT_EQ(h_upsampleExpected.size(), h_upsampleResults.size());

    
    for(size_t i = 0; i < h_upsampleResults.size(); i++)
    {
        // printf("h_upsampleResults[%lu]: %f, h_upsampleExpected[%lu]: %f\n", 
        //     i, h_upsampleResults[i],
        //     i, h_upsampleExpected[i]);
        ASSERT_FLOAT_EQ(h_upsampleResults[i], h_upsampleExpected[i]);
        //EXPECT_NEAR(h_upsampleResults[i], h_upsampleExpected[i], 1e-6);
        
    }

    upSample::cleanupDeviceMemory(d_Input);







    {
        printf("Read file\n");
        std::vector<float> h_filter = readFile("./vectors/interpFilter2.bin");
        ASSERT_NE(h_filter.size(), 0u);
        printf("h_filter.size: %lu\n", h_filter.size());

        auto[d_Output, d_Input, d_Filter] = interpolate::allocateDeviceMemory(h_data.size(), upsampleFactor, h_filter.size());

        interpolate::execute(d_Output,d_Input, h_data.size(), upsampleFactor);

        interpolate::cleanupDeviceMemory(d_Output, d_Input, d_Filter);
    }


}