#include <stdio.h>
#include <cstdlib>
#include <npp.h>
#include "cublas_v2.h"
#include <vector>
#include <numeric>
#include "interpolate.cuh"
#include "upsample.cuh"
#include "cuMemory.cuh"

namespace interpolate {

void execute(float * h_Output,
             float *d_Output, 
             const float *d_Input,
             int numElements,
             int upsampleFactor,
             const float *d_Filter,
             size_t const d_Filter_Length,
             float *d_Aux_Buffer);

std::tuple<float* /*d_Output*/, float* /*d_Input*/, float* /*d_Filter*/, float* /*d_Aux_Buffer*/> allocateDeviceMemory(size_t const numberOfSamples,
                                                                                              size_t const upsampleFactor,
                                                                                              size_t const filterLength)
{
    // printf("interpolate::allocateDeviceMemory, numberOfSamples:  %lu\n", numberOfSamples);
    // printf("interpolate::allocateDeviceMemory, upsampleFactor:   %lu\n", upsampleFactor);
    // printf("interpolate::allocateDeviceMemory, filterLength:     %lu\n", filterLength);

    // Allocate the memory for the device Input
    float *d_Input = NULL;
    {
        const size_t numberOfElements = numberOfSamples;
        //printf("interpolate::allocateDeviceMemory, d_Input, numberOfElements: %lu\n", numberOfElements);
        CHECK(cudaMalloc((void **)&d_Input, numberOfElements*sizeof(float)));
    }

    // Allocate the memory for the device Output
    float *d_Output = NULL;
    {
        const size_t numberOfElements = numberOfSamples*upsampleFactor + ((filterLength-1)*2);
        //printf("interpolate::allocateDeviceMemory, d_Output, numberOfElements: %lu\n", numberOfElements);
        CHECK(cudaMalloc((void **)&d_Output, numberOfElements*sizeof(float)));
    }

    float *d_Filter = NULL;
    {
        //const size_t numberOfElements = filterLength;
        //printf("interpolate::allocateDeviceMemory, d_Filter, numberOfElements: %lu\n", numberOfElements);
        CHECK(cudaMalloc((void **)&d_Filter, filterLength*sizeof(float)));
    }

    float *d_Aux_Buffer = NULL;
    {
        const size_t numberOfElements = filterLength;
        //printf("interpolate::allocateDeviceMemory, d_Aux_Buffer, numberOfElements: %lu\n", numberOfElements);
        CHECK(cudaMalloc((void **)&d_Aux_Buffer, numberOfElements*sizeof(float)));
    }
    return {d_Output, d_Input, d_Filter, d_Aux_Buffer};
}

void cleanupDeviceMemory(float * d_Output, float* d_Input, float* d_Filter, float* d_Aux_Buffer)
{
    CHECK(cudaFree(d_Output));
    CHECK(cudaFree(d_Input));
    CHECK(cudaFree(d_Filter));
    CHECK(cudaFree(d_Aux_Buffer));
}

void execute(float *h_Output,
             float *d_Output, 
             const float *d_Input,
             int numElements,
             int upsampleFactor,
             const float *d_Filter,
             size_t const d_Filter_Length,
             float *d_Aux_Buffer)
{
    //printf("interpolate::execute, numElements: %d, upsampleFactor: %d\n", numElements, upsampleFactor);

    size_t destIdx = d_Filter_Length - 1;
    //printf("interpolate::execute, upsample destIdx: %lu\n", destIdx);

    upSample::execute(&d_Output[destIdx], d_Input, numElements, upsampleFactor);

    std::vector<float> h_upsample(numElements*upsampleFactor);
    CHECK(cudaMemcpy(h_upsample.data(), d_Output, h_upsample.size()*sizeof(float), cudaMemcpyDeviceToHost));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    size_t endSize = (d_Filter_Length - 1) + (numElements * upsampleFactor) + (d_Filter_Length - 1);
    cuMemory<float> d_filtered_output(endSize);

    std::vector<float> h_partialMult(d_Filter_Length);
    //printf("interpolate::execute, endSize: %lu\n", endSize);
    for(size_t srcIdx = 0; srcIdx < endSize; srcIdx++)
    {
        cublasStatus_t stat = cublasSdot (handle, d_Filter_Length, d_Filter, 1, &d_Output[srcIdx], 1, d_filtered_output.data()+srcIdx);
    }
    cublasDestroy(handle);

    //printf("Copy the filtered output data from the CUDA device to the host memory\n");
    const size_t source_index = 0; //((d_Filter_Length-1)/2)*2; // Handle the pre-pending of zeros.  Handle the group delay of the FIR filter
    CHECK(cudaMemcpy(h_Output, d_filtered_output.data() + source_index, numElements*upsampleFactor*sizeof(float), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
}

} //namespace interpolate