#include "example.hpp"

#include <npp.h>
#include <vector>
#include <time.h>
#include <numeric>
#include <limits>

using namespace std;

int main(int argc, char* argv[]) {
  printf("nppGetGpuNumSMs: %d\n", nppGetGpuNumSMs());
  printf("nppGetMaxThreadsPerBlock: %d\n", nppGetMaxThreadsPerBlock());
  printf("nppGetMaxThreadsPerSM: %d\n", nppGetMaxThreadsPerSM());
  printf("nppGetGpuName: %s\n", nppGetGpuName());



  float elapsedTime;

  // pSrc, pSum, pDeviceBuffer are all device pointers. 
  Npp32f * pSrc; 
  Npp32f * pSum; 
  Npp8u * pDeviceBuffer;
  size_t nLength = 17e6;

  // Allocate the device memory.
  cudaMalloc((void **)(&pSrc), sizeof(Npp32f) * nLength);
  nppsSet_32f(1.0f, pSrc, nLength);  
  cudaMalloc((void **)(&pSum), sizeof(Npp32f) * 1);

  // Compute the appropriate size of the scratch-memory buffer, note that nBufferSize and nLength data types have changed from int to size_t. 
  int nBufferSize;
  nppsSumGetBufferSize_32f(nLength, &nBufferSize);
  // Allocate the scratch buffer 
  cudaMalloc((void **)(&pDeviceBuffer), nBufferSize);

  // Call the primitive with the scratch buffer
  clock_t start = clock();
  nppsSum_32f(pSrc, nLength, pSum, pDeviceBuffer);
  cudaDeviceSynchronize();
  elapsedTime = ((double) clock() - start) / CLOCKS_PER_SEC;
  Npp32f nSumHost;
  cudaMemcpy(&nSumHost, pSum, sizeof(Npp32f) * 1, cudaMemcpyDeviceToHost);
  printf("GPU time elapsed: %f seconds \n", elapsedTime);
  printf("sum = %f\n", nSumHost); // nSumHost = 1024.0f;

  // Free the device memory
  cudaFree(pSrc);
  cudaFree(pDeviceBuffer);
  cudaFree(pSum);

  {
    vector<Npp32f> vectSignal(nLength, 1);
    clock_t start = clock();
    Npp32f sum = 0;
    for(auto v : vectSignal)
    {
      sum += v; 
      if(sum >16777214)
      {
          sum = sum + 0.0;
      }
    }
    //sum = std::accumulate(vectSignal.begin(), vectSignal.end(), 0);
    elapsedTime = ((double) clock() - start) / CLOCKS_PER_SEC;
    printf("vectSignal.size: %lu\n", vectSignal.size());
    printf("max value for Npp32f: %f\n", std::numeric_limits<Npp32f>::max());
    printf("CPU time elapsed: %f seconds \n", elapsedTime);
    printf("sum = %f\n", sum); // nSumHost = 1024.0f;

  }



  return 0;
}