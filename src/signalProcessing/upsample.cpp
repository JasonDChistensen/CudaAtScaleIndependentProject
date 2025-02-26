#include <stdio.h>
#include <cstdlib>
#include "upsample.hh"

float * allocateDeviceMemory(size_t numberOfSamples, size_t upsampleFactor)
{
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Allocate the device output vector U
  float *d_U = NULL;
  err = cudaMalloc((void **)&d_U, numberOfSamples*upsampleFactor*sizeof(float));

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
      cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return d_U;
}

void cleanupDeviceMemory(float * d_U)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Free device global memory
    err = cudaFree(d_U);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
