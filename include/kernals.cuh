
void hostUpsample(float *d_Output, const float *d_Input, int numElements, int upsampleFactor);

__global__ void deviceUpsample(float *output, const float *input, int numElements, int upsampleFactor);