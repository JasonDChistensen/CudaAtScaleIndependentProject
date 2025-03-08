// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <stddef.h>

namespace upSample {

float * allocateDeviceMemory(size_t numberOfSamples, size_t upsampleFactor);
void cleanupDeviceMemory(float * d_U);

void execute(float *d_Output, const float *d_Input, int numElements, int upsampleFactor);

}