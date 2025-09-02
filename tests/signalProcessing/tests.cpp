#include <gtest/gtest.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "readFile.hh"
#include "upsample.cuh"
#include "interpolate.cuh"
#include "cuMemory.cuh"

#include <npp.h>  //TODO: Remove


// Define a test fixture class for cuBlas tests
class cuBlasTestFixture : public ::testing::Test {
protected:
    // SetUp method to initialize resources before each test
    void SetUp() override {
        cublasStatus_t stat = cublasCreate(&fixture_handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("cuBlasTestFixture initialization failed\n");
            exit(EXIT_FAILURE);
        }
  }

  // TearDown method to release resources after each test
  void TearDown() override {
     cublasDestroy(fixture_handle);
  }

  // Shared resource accessible to all tests in the fixture
  cublasHandle_t fixture_handle;
};







TEST(SIGNAL_PROCESSING, DISABLED_NPP_MULTIPLY_EXAMPLE)
{
    size_t nLength = 100;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate the memory for Source 1
    float *pSrc1 = NULL;
    CHECK(cudaMalloc((void **)&pSrc1, nLength*sizeof(Npp32f)));

    // Allocate the memory for Source 2
    float *pSrc2 = NULL;
    CHECK(cudaMalloc((void **)&pSrc2, nLength*sizeof(Npp32f)));

    // Allocate the memory for Destination
    float *pDst = NULL;
    CHECK(cudaMalloc((void **)&pDst, nLength*sizeof(Npp32f)));

    CHECK_NPP(nppsSet_32f(2.0, pSrc1, nLength));
    CHECK_NPP(nppsSet_32f(3.0, pSrc2, nLength));
    CHECK_NPP(nppsMul_32f(pSrc1, pSrc2, pDst, nLength));


    std::vector<Npp32f> h_result(nLength);
    CHECK(cudaMemcpy(h_result.data(), pDst, h_result.size()*sizeof(Npp32f), cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < 30; i++)
    {
        printf("h_result[%lu]: %f\n", i, h_result[i]);
    }

}

/* This tests the cuBlas dot product */
TEST_F(cuBlasTestFixture, CUBLAS_DOT_PRODUCT)
{
    size_t nLength = 100;

    cuMemory<Npp32f> src1(nLength);
    cuMemory<Npp32f> src2(nLength);
    cuMemory<Npp32f> dst(1);

    const Npp32f SRC1_INIT = 1.0;
    const Npp32f SRC2_INIT = 2.0;
    const Npp32f EXECTED_OUTPUT = SRC1_INIT*SRC2_INIT*nLength;

    CHECK_NPP(nppsSet_32f(SRC1_INIT, src1.data(), src1.size()));
    CHECK_NPP(nppsSet_32f(SRC2_INIT, src2.data(), src2.size()));
    CHECK_CUBLAS(cublasSdot (fixture_handle, src1.size(), src1.data(), 1, src2.data(), 1, dst.data()));

    EXPECT_EQ(dst.get_data().front(), EXECTED_OUTPUT);
    //printf("dst result: %f\n", dst.get_data().front());
}

TEST(SIGNAL_PROCESSING, DISABLED_CU_MEMORY)
{
    cuMemory<float> src0(0);        
    EXPECT_EQ(src0.data(), nullptr);
    EXPECT_EQ(src0.size(), 0);

    cuMemory<float> src1(1);   
    EXPECT_NE(src1.data(), nullptr);
    EXPECT_EQ(src1.size(), 1);

    cuMemory<float> src100(100);   
    EXPECT_NE(src100.data(), nullptr);
    EXPECT_EQ(src100.size(), 100);
}



TEST(SIGNAL_PROCESSING, DISABLED_UPSAMPLE)
{
    std::vector<float> h_Input = readFile("./vectors/inputSignal2.bin");
    ASSERT_NE(h_Input.size(), 0u);
    printf("h_Input.size: %lu\n", h_Input.size());


    const size_t upsampleFactor  = 2;
    float* d_Input = upSample::allocateDeviceMemory(h_Input.size(), upsampleFactor);
    ASSERT_NE(d_Input, nullptr);
    printf("d_Input: %p\n", d_Input);
    CHECK(cudaMemcpy(d_Input, h_Input.data(), h_Input.size()*sizeof(float), cudaMemcpyHostToDevice));

    float* d_Output = upSample::allocateDeviceMemory(h_Input.size(), upsampleFactor);
    ASSERT_NE(d_Output, nullptr);
    printf("d_Output: %p\n", d_Output);

    int numElements = h_Input.size();
    upSample::execute(d_Output, d_Input, numElements, upsampleFactor);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    std::vector<float> h_Expected_Output = readFile("./vectors/inputSignalUpsampled2.bin");
    ASSERT_NE(h_Expected_Output.size(), 0u);
    printf("h_Expected_Output.size: %lu\n", h_Expected_Output.size());


    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    std::vector<float> h_Output(h_Input.size()*upsampleFactor);

    CHECK(cudaMemcpy(h_Output.data(), d_Output, h_Output.size()*sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_EQ(h_Input.size()*upsampleFactor, h_Output.size());
    ASSERT_EQ(h_Expected_Output.size(), h_Output.size());

    
    for(size_t i = 0; i < h_Output.size(); i++)
    {
        // printf("h_Output[%lu]: %f, h_Expected_Output[%lu]: %f\n",
        //     i, h_Output[i],i, h_Expected_Output[i]);
        ASSERT_FLOAT_EQ(h_Output[i], h_Expected_Output[i]);      
    }

    upSample::cleanupDeviceMemory(d_Input);
}

TEST(SIGNAL_PROCESSING, DISABLED_INTERPOLATE)
{
    std::vector<float> h_Input = readFile("./vectors/inputSignal2.bin");
    ASSERT_NE(h_Input.size(), 0u);
    printf("h_Input.size: %lu\n", h_Input.size());

    std::vector<float> h_filter = readFile("./vectors/interpFilter2.bin");
    ASSERT_NE(h_filter.size(), 0u);
    printf("h_filter.size: %lu\n", h_filter.size());
    for(size_t i = 0; i < h_filter.size(); i++)
    {
        printf("h_filter[%lu]: %f\n", i, h_filter[i]);
    }
    cuMemory<float> d_Filter(h_filter);
    cuMemory<float> d_Aux_Buffer(h_filter.size());

    const size_t upsampleFactor  = 2;
    const size_t numberOfOutputElements = h_Input.size()*upsampleFactor + ((d_Filter.size()-1)*2);
    cuMemory<float> d_Output(numberOfOutputElements);


    printf("Read the Upsampled results file\n");
    std::vector<float> h_matlabInterpolatedOutput = readFile("./vectors/matlabInterpolatedOutput2.bin");
    ASSERT_NE(h_matlabInterpolatedOutput.size(), 0u);
    printf("h_matlabInterpolatedOutput.size: %lu\n", h_matlabInterpolatedOutput.size());


    cuMemory<float> d_Input(numberOfOutputElements);
    CHECK(cudaMemcpy(d_Input.data(), h_Input.data(), h_Input.size()*sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_Output(h_Input.size()*upsampleFactor);
    printf("h_Output.size: %lu\n", h_Output.size());
    interpolate::execute(h_Output.data(), d_Output.data(), d_Input.data(), h_Input.size(), upsampleFactor, 
                         d_Filter.data(), h_filter.size(),d_Aux_Buffer.data());


    // std::vector<float> h_Output(h_Input.size()*upsampleFactor);
    // err = cudaMemcpy(h_Output.data(), d_Output, h_Output.size()*sizeof(float), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) {
    //     fprintf(stderr,
    //     "Failed to copy d_Output vector from device to host (error code %s)!\n",
    //     cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    for(size_t i = 0; i < 30; i++)
    {
        printf("h_Output[%lu]: %f, h_matlabInterpolatedOutput[%lu]: %f\n", i, h_Output[i], i, h_matlabInterpolatedOutput[i]);
    }

    ASSERT_EQ(h_matlabInterpolatedOutput.size(), h_Output.size());
    for(size_t i = 0; i < h_matlabInterpolatedOutput.size(); i++)
    {
        //printf("h_Output[%lu]: %f\n", i, h_Output[i]);
        ASSERT_NEAR(h_matlabInterpolatedOutput[i], h_Output[i], 1e-6);
    }
 }
