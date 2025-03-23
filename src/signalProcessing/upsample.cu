#include <stdio.h>
#include <cstdlib>
#include "upsample.cuh"

namespace upSample {

void execute(float *d_Output, const float *d_Input, int numElements, int upsampleFactor);

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

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void deviceUpsample(float *output, const float *input, int numElements, int upsampleFactor)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        int startingIndex = i*upsampleFactor;
        output[startingIndex] = input[i];

        for(int i = 1; i < upsampleFactor; i++)
        {
            output[startingIndex+i] = 0;
        }
   }
}

void execute(float *d_Output, const float *d_Input, int numElements, int upsampleFactor)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    deviceUpsample<<<blocksPerGrid, threadsPerBlock>>>(d_Output, d_Input, numElements, upsampleFactor);
}

} //namespace upSample 