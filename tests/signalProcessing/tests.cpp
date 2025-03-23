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
    //Nothing for now
  }

  // Shared resource accessible to all tests in the fixture
  cublasHandle_t fixture_handle;
};


TEST(SIGNAL_PROCESSING, UPSAMPLE)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    printf("Read file\n");
    std::vector<float> h_data = readFile("./vectors/inputSignal2.bin");
    ASSERT_NE(h_data.size(), 0u);
    printf("h_data.size: %lu\n", h_data.size());


    size_t upsampleFactor  = 2;
    float* d_Input = upSample::allocateDeviceMemory(h_data.size(), upsampleFactor);
    ASSERT_NE(d_Input, nullptr);
    printf("d_Input: %p\n", d_Input);
    err = cudaMemcpy(d_Input, h_data.data(), h_data.size()*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector from host to device (error code %s)!\n",
            cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    float* d_Output = upSample::allocateDeviceMemory(h_data.size(), upsampleFactor);
    ASSERT_NE(d_Output, nullptr);
    printf("d_Output: %p\n", d_Output);



    int numElements = h_data.size();
    upSample::execute(d_Output, d_Input, numElements, upsampleFactor);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // // Copy the device result vector in device memory to the host result vector
    // // in host memory.
    // printf("Copy output data from the CUDA device to the host memory\n");
    // std::vector<float> h_results(h_data.size());

    // err = cudaMemcpy(h_results.data(), d_Output, h_data.size()*sizeof(float), cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) {
    //     fprintf(stderr,
    //         "Failed to copy vector C from device to host (error code %s)!\n",
    //         cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // ASSERT_EQ(h_data.size(), h_results.size());

    // for(size_t i = 0; i < h_data.size(); i++)
    // {
    //     EXPECT_FLOAT_EQ(h_data[i], h_results[i]);
    // }


    printf("Read file\n");
    std::vector<float> h_upsampleExpected = readFile("./vectors/inputSignalUpsampled2.bin");
    ASSERT_NE(h_upsampleExpected.size(), 0u);
    printf("h_upsampleExpected.size: %lu\n", h_upsampleExpected.size());





    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    std::vector<float> h_upsampleResults(h_data.size()*upsampleFactor);

    err = cudaMemcpy(h_upsampleResults.data(), d_Output, h_upsampleResults.size()*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
            "Failed to copy d_Output vector from device to host (error code %s)!\n",
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    ASSERT_EQ(h_data.size()*upsampleFactor, h_upsampleResults.size());
    ASSERT_EQ(h_upsampleExpected.size(), h_upsampleResults.size());

    
    for(size_t i = 0; i < h_upsampleResults.size(); i++)
    {
        // printf("h_upsampleResults[%lu]: %f, h_upsampleExpected[%lu]: %f\n", 
        //     i, h_upsampleResults[i],
        //     i, h_upsampleExpected[i]);
        ASSERT_FLOAT_EQ(h_upsampleResults[i], h_upsampleExpected[i]);
        //EXPECT_NEAR(h_upsampleResults[i], h_upsampleExpected[i], 1e-6);
        
    }

    upSample::cleanupDeviceMemory(d_Input);







    {
        printf("Read the Filter file\n");
        std::vector<float> h_filter = readFile("./vectors/interpFilter2.bin");
        ASSERT_NE(h_filter.size(), 0u);
        printf("h_filter.size: %lu\n", h_filter.size());
        for(size_t i = 0; i < h_filter.size(); i++)
        {
            printf("h_filter[%lu]: %f\n", i, h_filter[i]);
        }


        auto[d_Output, d_Input, d_Filter] = interpolate::allocateDeviceMemory(h_data.size(), upsampleFactor, h_filter.size());


        err = cudaMemcpy(d_Filter, h_filter.data(), h_filter.size()*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr,
                "Failed to copy h_filter vector from d_Filter (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpy(d_Input, h_data.data(), h_data.size()*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr,
                "Failed to copy h_data vector from d_Input (error code %s)!\n",
                cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        std::vector<float> h_output(h_data.size()*upsampleFactor + (h_filter.size() -1)*2);
        printf("h_output.size: %lu\n", h_output.size());
        interpolate::execute(h_output.data(), d_Output, d_Input, h_data.size(), upsampleFactor, d_Filter, h_filter.size());

        // std::vector<float> h_output(h_data.size()*upsampleFactor);
        // err = cudaMemcpy(h_output.data(), d_Output, h_output.size()*sizeof(float), cudaMemcpyDeviceToHost);
        // if (err != cudaSuccess) {
        //     fprintf(stderr,
        //     "Failed to copy d_Output vector from device to host (error code %s)!\n",
        //     cudaGetErrorString(err));
        //     exit(EXIT_FAILURE);
        // }

        for(size_t i = 0; i < 30; i++)
        {
            printf("h_output[%lu]: %f\n", i, h_output[i]);
        }

        interpolate::cleanupDeviceMemory(d_Output, d_Input, d_Filter);
    }


}

TEST(SIGNAL_PROCESSING, DISABLED_NPP_MULTIPLY_EXAMPLE)
{
    size_t nLength = 100;
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate the memory for Source 1
    float *pSrc1 = NULL;
    err = cudaMalloc((void **)&pSrc1, nLength*sizeof(Npp32f));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for Source 1 (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the memory for Source 2
    float *pSrc2 = NULL;
    err = cudaMalloc((void **)&pSrc2, nLength*sizeof(Npp32f));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for Source 2 (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the memory for Destination
    float *pDst = NULL;
    err = cudaMalloc((void **)&pDst, nLength*sizeof(Npp32f));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for Destination (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    NppStatus status = nppsSet_32f(2.0, pSrc1, nLength);
    status = nppsSet_32f(3.0, pSrc2, nLength);
    status = nppsMul_32f(pSrc1, pSrc2, pDst, nLength);


    std::vector<Npp32f> h_result(nLength);
    err = cudaMemcpy(h_result.data(), pDst, h_result.size()*sizeof(Npp32f), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
        "Failed to copy d_Output vector from device to host (error code %s)!\n",
        cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

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

    NppStatus status = nppsSet_32f(SRC1_INIT, src1.get_ptr(), src1.get_length());
    status = nppsSet_32f(SRC2_INIT, src2.get_ptr(), src2.get_length());
    cublasStatus_t stat = cublasSdot (fixture_handle, src1.get_length(), src1.get_ptr(), 1, src2.get_ptr(), 1, dst.get_ptr());

    EXPECT_EQ(dst.get_data().front(), EXECTED_OUTPUT);
    printf("dst result: %f\n", dst.get_data().front());
}

TEST(SIGNAL_PROCESSING, CU_MEMORY)
{
    cuMemory<float> src0(0);        
    EXPECT_EQ(src0.get_ptr(), nullptr);
    EXPECT_EQ(src0.get_length(), 0);

    cuMemory<float> src1(1);   
    EXPECT_NE(src1.get_ptr(), nullptr);
    EXPECT_EQ(src1.get_length(), 1);

    cuMemory<float> src100(100);   
    EXPECT_NE(src100.get_ptr(), nullptr);
    EXPECT_EQ(src100.get_length(), 100);
}