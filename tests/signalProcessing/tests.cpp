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
            printf ("cuBlasTestFixture initialization failed:%d\n", stat);
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






/* This is a toy example using nppsMul_32f.
 * It will multiply a vector of 2's by a vector of 3's
 * with the expected output being 6's.
 */
TEST_F(cuBlasTestFixture, NPP_MULTIPLY_EXAMPLE)
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
        ASSERT_FLOAT_EQ(6.0, h_result[i]); 
    }

    CHECK(cudaFree(pSrc1));
    CHECK(cudaFree(pSrc2));
    CHECK(cudaFree(pDst));

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

