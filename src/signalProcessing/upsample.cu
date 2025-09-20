#include <stdio.h>
#include <cstdlib>
#include "upsample.cuh"

namespace upSample {

void execute(float *d_Output, float const *d_Input, int numElements, int upsampleFactor);

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
__global__ void deviceUpsample(float *output, float const *input, int numElements, int upsampleFactor)
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

void execute(float *d_Output, float const *d_Input, int numElements, int upsampleFactor)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0); // 0 indicates the default stream

    deviceUpsample<<<blocksPerGrid, threadsPerBlock>>>(d_Output, d_Input, numElements, upsampleFactor);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Synchronize the stop event to ensure all preceding operations in the stream are complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //printf("upSample Kernel execution time: %f ms\n", milliseconds);

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

} //namespace upSample 