#include <stdio.h>
#include <cstdlib>
#include <npp.h>
#include <vector>
#include <numeric>
#include "interpolate.cuh"
#include "upsample.cuh"

namespace interpolate {

void execute(float *h_Output, float *d_Output, const float *d_Input, int numElements, int upsampleFactor, const float *d_Filter, size_t const filterLength);

std::tuple<float* /*d_Output*/, float* /*d_Input*/, float* /*d_Filter*/> allocateDeviceMemory(size_t const numberOfSamples,
                                                                                              size_t const upsampleFactor,
                                                                                              size_t const filterLength)
{
  printf("interpolate::allocateDeviceMemory, numberOfSamples:  %lu\n", numberOfSamples);
  printf("interpolate::allocateDeviceMemory, upsampleFactor:   %lu\n", upsampleFactor);
  printf("interpolate::allocateDeviceMemory, filterLength:     %lu\n", filterLength);

  // Allocate the memory for the device Input
  float *d_Input = NULL;
  {
    const size_t numberOfElements = numberOfSamples;
    printf("interpolate::allocateDeviceMemory, d_Input, numberOfElements: %lu\n", numberOfElements);
    cudaError_t err = cudaMalloc((void **)&d_Input, numberOfElements*sizeof(float));
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
    cudaError_t err = cudaMalloc((void **)&d_Output, numberOfElements*sizeof(float));
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
    cudaError_t err = cudaMalloc((void **)&d_Filter, filterLength*sizeof(float));
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
    // Free device global memory
    {
      cudaError_t err = cudaFree(d_Output);
      if (err != cudaSuccess) {
          fprintf(stderr, "Failed to free d_Output (error code %s)!\n",
          cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }
    }

    {
      cudaError_t err = cudaFree(d_Input);
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free d_Input (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
    }

    {
      cudaError_t err = cudaFree(d_Filter);
      if (err != cudaSuccess) {
          fprintf(stderr, "Failed to free d_Filter (error code %s)!\n",
          cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }
    }
}

void execute(float *h_Output, float *d_Output, const float *d_Input, int numElements, int upsampleFactor, const float *d_Filter, size_t const filterLength)
{
  printf("interpolate::execute\n");
  printf("interpolate::execute, numElements:      %d\n", numElements);
  printf("interpolate::execute, upsampleFactor:   %d\n", upsampleFactor);

  size_t destIdx = filterLength - 1;
  printf("interpolate::execute, upsample destIdx: %lu\n", destIdx);

  upSample::execute(&d_Output[destIdx], d_Input, numElements, upsampleFactor);

  std::vector<float> h_upsample(numElements*upsampleFactor);
  cudaError_t err = cudaMemcpy(h_upsample.data(), d_Output, h_upsample.size()*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy d_Output vector from device to host (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
  }
  for(size_t i = 0; i < 30; i++)
  {
      printf("interpolate::execute, h_upsample[%lu]: %f\n", i, h_upsample[i]);
  }



  float *d_PartialMult = NULL;
  {
    const size_t numberOfElements = filterLength;
    printf("interpolate::allocateDeviceMemory, d_Filter, numberOfElements: %lu\n", numberOfElements);
    cudaError_t err = cudaMalloc((void **)&d_PartialMult, filterLength*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_Filter (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  }



  size_t endSize = (filterLength - 1) + (numElements * upsampleFactor) + (filterLength - 1);
  std::vector<float> h_partialMult(filterLength);
  printf("interpolate::execute, endSize: %lu\n", endSize);
  for(size_t srcIdx = 0; srcIdx < endSize; srcIdx++)
  {

    printf("interpolate::execute, srcIdx: %lu\n", srcIdx);
    NppStatus status = nppsMul_32f(d_Filter, &d_Output[srcIdx], d_PartialMult, filterLength);
    if (status != NPP_SUCCESS) {
        fprintf(stderr, "nppsMul_32f Failed!\n");
        exit(EXIT_FAILURE);
    }

    //printf("Copy output data from the CUDA device to the host memory\n");
    cudaError_t err = cudaMemcpy(h_partialMult.data(), d_PartialMult, h_partialMult.size()*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy d_Output vector from device to host (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // for(size_t i = 0; i < h_partialMult.size(); i++)
    // {
    //   printf("interpolate::execute, h_partialMult[%lu]: %f\n", i, h_partialMult[i]);
    // }


    float sum = std::accumulate(h_partialMult.begin(), h_partialMult.end(), 0.0);
    printf("sum: %1.10f\n", sum);
    h_Output[srcIdx] = sum;
  }


  {
    cudaError_t err = cudaFree(d_PartialMult);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free d_PartialMult (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  }

  printf("nppGetGpuNumSMs: %d\n", nppGetGpuNumSMs());
  printf("nppGetMaxThreadsPerBlock: %d\n", nppGetMaxThreadsPerBlock());
  printf("nppGetMaxThreadsPerSM: %d\n", nppGetMaxThreadsPerSM());
  printf("nppGetGpuName: %s\n", nppGetGpuName());

}

} //namespace interpolate