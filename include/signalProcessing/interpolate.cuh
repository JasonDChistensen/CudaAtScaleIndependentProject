// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <stddef.h>
#include <tuple>

namespace interpolate {

std::tuple<float* /*d_Output*/, float* /*d_Input*/, float* /*d_Filter*/> allocateDeviceMemory(size_t const numberOfSamples,
                                                                                              size_t const upsampleFactor,
                                                                                              size_t const filterLength);
void cleanupDeviceMemory(float * d_Output, float* d_Input, float* d_Filter);

void execute(float *h_Output, float *d_Output, const float *d_Input, int numElements, int upsampleFactor, const float *d_Filter, size_t const filterLength);

}