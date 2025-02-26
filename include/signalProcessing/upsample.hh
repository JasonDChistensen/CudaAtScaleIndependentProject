// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <stddef.h>

float * allocateDeviceMemory(size_t numberOfSamples, size_t upsampleFactor);

void cleanupDeviceMemory(float * d_U);