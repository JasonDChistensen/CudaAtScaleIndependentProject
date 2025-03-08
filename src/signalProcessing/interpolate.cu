#include <stdio.h>
#include <cstdlib>
#include <npp.h>
#include "interpolate.cuh"

namespace interpolate {

void execute(float *d_Output, const float *d_Input, int numElements, int upsampleFactor);

std::tuple<float* /*d_Output*/, float* /*d_Input*/, float* /*d_Filter*/> allocateDeviceMemory(size_t const numberOfSamples,
                                                                                              size_t const upsampleFactor,
                                                                                              size_t const filterLength)
{
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  printf("interpolate::allocateDeviceMemory, numberOfSamples:  %lu\n", numberOfSamples);
  printf("interpolate::allocateDeviceMemory, upsampleFactor:   %lu\n", upsampleFactor);
  printf("interpolate::allocateDeviceMemory, filterLength:     %lu\n", filterLength);

  // Allocate the memory for the device Input
  float *d_Input = NULL;
  {
    const size_t numberOfElements = numberOfSamples;
    printf("interpolate::allocateDeviceMemory, d_Input, numberOfElements: %lu\n", numberOfElements);
    err = cudaMalloc((void **)&d_Input, numberOfElements*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  }

  // Allocate the memory for the device Output
  float *d_Output = NULL;
  {
    const size_t numberOfElements = numberOfSamples*upsampleFactor + ((filterLength-1)*2);
    printf("interpolate::allocateDeviceMemory, d_Output, numberOfElements: %lu\n", numberOfElements);
    err = cudaMalloc((void **)&d_Output, numberOfElements*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  }

  float *d_Filter = NULL;
  {
    const size_t numberOfElements = filterLength;
    printf("interpolate::allocateDeviceMemory, d_Filter, numberOfElements: %lu\n", numberOfElements);
    err = cudaMalloc((void **)&d_Filter, filterLength*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_Filter (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  }


  return {d_Output, d_Input, d_Filter};
}

void cleanupDeviceMemory(float * d_Output, float* d_Input, float* d_Filter)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Free device global memory
    err = cudaFree(d_Output);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free d_Output (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_Input);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free d_Input (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_Filter);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free d_Filter (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void execute(float *d_Output, const float *d_Input, int numElements, int upsampleFactor)
{

  printf("interpolate::execute\n");
  printf("nppGetGpuNumSMs: %d\n", nppGetGpuNumSMs());
  printf("nppGetMaxThreadsPerBlock: %d\n", nppGetMaxThreadsPerBlock());
  printf("nppGetMaxThreadsPerSM: %d\n", nppGetMaxThreadsPerSM());
  printf("nppGetGpuName: %s\n", nppGetGpuName());

}

} //namespace interpolate