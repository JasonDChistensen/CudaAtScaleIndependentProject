#include <npp.h>
#include <vector>
#include <time.h>
#include <numeric>
#include <limits>
//#include <cstdio>  //for printf
#include <iostream> //for cout
#include <string>
#include <set>
#include <map>
#include <cmath> // Required for std::fabs and std::abs
#include "readFile.hh"
#include "writeFile.hh"
#include "cuMemory.cuh"
#include "interpolate.cuh"

using namespace std;
#define INTERPOLATION_FACTOR int32_t
#define FILTER_FILE_NAME std::string

static const std::set<int32_t> valid_interpolation_factors = { 2, 3, 5, 7};
static const std::map<INTERPOLATION_FACTOR, FILTER_FILE_NAME> interpolation_factor_to_file_name = 
{
  { 2, "./tests/vectors/interpFilter2.bin"},
  { 3, "./tests/vectors/interpFilter3.bin"},
  { 5, "./tests/vectors/interpFilter5.bin"},
  { 7, "./tests/vectors/interpFilter7.bin"}
};

void print_help_message(void)
{
  cout << "This application will perform interpolation on the input file provided."  << endl;
  cout << "The input file shoud be an array of floats.  The output will be an array of floats" << endl;
  cout << "The supported interpolation factors are 2, 3, 5, and 7." << endl;
  cout << "Usage: cudaAtScaleIndependentProject.exe <output file> <input file> <interpolation factor>" << endl;

  cout << endl;
  cout << "nppGetGpuNumSMs:" << nppGetGpuNumSMs() << endl;
  cout << "nppGetMaxThreadsPerBlock:" << nppGetMaxThreadsPerBlock() << endl;
  cout << "nppGetMaxThreadsPerSM:" << nppGetMaxThreadsPerSM() << endl;
  cout << "nppGetGpuName:" << nppGetGpuName() << endl;
}

bool string_to_int(int32_t& out, char * in)
{
  bool status = false;
  try {
    out = std::stoi(in);
    status = true;
  } catch (const std::invalid_argument& e) {
    std::cerr << "Error: Invalid argument - Not a valid integer format." << std::endl;
  } catch (const std::out_of_range& e) {
    std::cerr << "Error: Out of range - Number is too large or too small." << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: An unexpected exception occurred: " << e.what() << std::endl;
  }
  return status;
}

int main(int argc, char* argv[])
{

  if(argc != 4)
  {
    print_help_message();
    return -1;
  }
  std::string input_file;
  std::string output_file;
  int32_t interpolation_factor;

  for (int i = 1; i < argc; ++i) 
  {
    std::string arg = argv[i];

    switch(i)
    {
      case 1:
        output_file = argv[i];
        break;
      case 2:
        input_file = argv[i];
        break;
      case 3:
        if(!string_to_int(interpolation_factor, argv[i]))
        {
          print_help_message();
          return -1;
        }
        if(valid_interpolation_factors.count(interpolation_factor) == 0)
        {
          cout << "interpolation factor " << interpolation_factor <<  " is not supported." << std::endl;
          print_help_message();
          return -1;
        }
        break;
      default:
        print_help_message();
        return -1;
    }
  }
  
  printf("output file:%s\n", output_file.c_str());
  printf("input file:%s\n", input_file.c_str());
  printf("interpolation factor:%d\n", interpolation_factor);


  /* Get the interpolation filter file name */
  std::string filter_file;
  if(interpolation_factor_to_file_name.count(interpolation_factor) > 0)
  {
    filter_file = interpolation_factor_to_file_name.at(interpolation_factor);
  }
  else
  {
    printf("Uh oh! Something went wrong.  Not able to find interpolation "
           "filter file for interpolation factor:%d!\n", interpolation_factor);
    return -1;
  }
  printf("interpolation filter file name:%s\n", filter_file.c_str());
  
  /* Read in the interpolation filter coeff's */
  std::vector<float> h_filter = readFile(filter_file);

  /* Read in the file to be interpolated*/
  // example: ./cudaAtScaleIndependentProject.exe  ./output2.bin ./tests/vectors/inputSignal2.bin 2
  std::vector<float> h_Input = readFile(input_file);

  // Allocate memory
  cuMemory<float> d_Filter(h_filter);
  cuMemory<float> d_Aux_Buffer(h_filter.size());

  const size_t upsampleFactor  = interpolation_factor;
  const size_t numberOfOutputElements = h_Input.size()*upsampleFactor + ((d_Filter.size()-1)*2);
  // printf("h_Input.size():%zu\n", h_Input.size());
  // printf("upsampleFactor:%zu\n", upsampleFactor);
  // printf("numberOfOutputElements:%zu\n", numberOfOutputElements);
  cuMemory<float> d_Output(numberOfOutputElements);

  cuMemory<float> d_Input(numberOfOutputElements);
  CHECK(cudaMemcpy(d_Input.data(), h_Input.data(), h_Input.size()*sizeof(float), cudaMemcpyHostToDevice));

  std::vector<float> h_Output(h_Input.size()*upsampleFactor);
  //printf("h_Output.size: %lu\n", h_Output.size());
  interpolate::execute(h_Output.data(), d_Output.data(), d_Input.data(), h_Input.size(), upsampleFactor, 
                       d_Filter.data(), h_filter.size(), d_Aux_Buffer.data());

  writeFile(output_file, h_Output);

  //printf("Read the Upsampled results file\n");
  std::vector<float> h_matlabInterpolatedOutput = readFile("./tests/vectors/matlabInterpolatedOutput2.bin");
  //printf("h_matlabInterpolatedOutput.size: %lu\n", h_matlabInterpolatedOutput.size());

  std::vector<float> output = readFile(output_file);

  if(h_matlabInterpolatedOutput.size() != output.size())
  {
    printf("[%s:%d]Error", __FILE__, __LINE__);
    exit(1);
  }

  printf("[%s:%d]output.size:%zu\n", __FILE__, __LINE__, output.size());

  for(size_t i = 0; i < h_matlabInterpolatedOutput.size(); i++)
  {
    float error = std::fabs(h_matlabInterpolatedOutput[i] - output[i]);
    if(error > 1e-5)
    {
      printf("[%s:%d]Error!, i:%zu,h_matlabInterpolatedOutput:%.12f, output:%.12f\n", __FILE__, __LINE__,
        i,
        h_matlabInterpolatedOutput[i],
        output[i]
      );
      exit(1);
    }
  }
                     
  cudaDeviceReset(); 

  return 0;
}