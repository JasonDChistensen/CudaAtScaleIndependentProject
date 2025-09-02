#include <gtest/gtest.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "readFile.hh"
#include "cuMemory.cuh"
#include "upsample.cuh"
#include "interpolate.cuh"

TEST(FACTOR3, INTERPOLATE)
{
    std::vector<float> h_Input = readFile("./vectors/inputSignal3.bin");
    ASSERT_NE(h_Input.size(), 0u);
    printf("h_Input.size: %lu\n", h_Input.size());

    std::vector<float> h_filter = readFile("./vectors/interpFilter3.bin");
    ASSERT_NE(h_filter.size(), 0u);
    printf("h_filter.size: %lu\n", h_filter.size());
    for(size_t i = 0; i < h_filter.size(); i++)
    {
        printf("h_filter[%lu]: %f\n", i, h_filter[i]);
    }
    cuMemory<float> d_Filter(h_filter);
    cuMemory<float> d_Aux_Buffer(h_filter.size());

    const size_t upsampleFactor  = 3;
    const size_t numberOfOutputElements = h_Input.size()*upsampleFactor + ((d_Filter.size()-1)*2);
    cuMemory<float> d_Output(numberOfOutputElements);


    printf("Read the Upsampled results file\n");
    std::vector<float> h_matlabInterpolatedOutput = readFile("./vectors/matlabInterpolatedOutput3.bin");
    ASSERT_NE(h_matlabInterpolatedOutput.size(), 0u);
    printf("h_matlabInterpolatedOutput.size: %lu\n", h_matlabInterpolatedOutput.size());


    cuMemory<float> d_Input(numberOfOutputElements);
    CHECK(cudaMemcpy(d_Input.data(), h_Input.data(), h_Input.size()*sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_Output(h_Input.size()*upsampleFactor);
    printf("h_Output.size: %lu\n", h_Output.size());
    interpolate::execute(h_Output.data(), d_Output.data(), d_Input.data(), h_Input.size(), upsampleFactor, 
                         d_Filter.data(), h_filter.size(),d_Aux_Buffer.data());

    for(size_t i = 0; i < 30; i++)
    {
        printf("h_Output[%lu]: %f, h_matlabInterpolatedOutput[%lu]: %f\n", i, h_Output[i], i, h_matlabInterpolatedOutput[i]);
    }

    ASSERT_EQ(h_matlabInterpolatedOutput.size(), h_Output.size());
    for(size_t i = 0; i < h_matlabInterpolatedOutput.size(); i++)
    {
        printf("h_Output[%lu]: %f\n", i, h_Output[i]);
        ASSERT_NEAR(h_matlabInterpolatedOutput[i], h_Output[i], 1e-6);
    }
 }