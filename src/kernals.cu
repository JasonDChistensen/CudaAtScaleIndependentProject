#include "kernals.cuh"

void hostUpsample(float *d_Output, const float *d_Input, int numElements, int upsampleFactor)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    //void deviceUpsample(const float *output, const float *input, int numElements, int upsampleFactor)
    deviceUpsample<<<blocksPerGrid, threadsPerBlock>>>(d_Output, d_Input, numElements, upsampleFactor);

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