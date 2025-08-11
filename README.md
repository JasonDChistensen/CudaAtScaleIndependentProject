# Cuda At Scale Independent Project

For this project I decided to implement interpolation using CUDA.  Why interpolation?  Interpolation is useful in its own right.  At the same time once interpolation has been implemented it is not to difficult to do decimation, correlation, and filtering.

I believe the intention of the project was to use the NPP libray.  My plan was to use nppsMul_32f to multiply the input by the interpolaton filter, then use nppsIntegral_32s_Ctx to accumulate the resullts.  The problem was that I needed an accumulation function that worked with float's and nppsIntegral_32s_Ctx only accepts 32-bit signed integers.  I could have made my own kernel to accumulate float's, but that not be in the spirit of using CUDA libraries.  As an alternative I used cublasSdot for the accumalation.

## Commands to make the project:

```
git submodule update --init --recursive
cd src
make
```
## Commands to run the executable:
```
cudaAtScaleIndependentProject.exe <output file> <input file> <interpolation factor>
```
## Commands to time the executable:
```
time cudaAtScaleIndependentProject.exe <output file> <input file> <interpolation factor>
```


## Commands to make the unit tests:

```
cd ../tests
make
```

## Commands to make run the unit tests:
```
newProjectTests.exe
```

## Reference:

https://docs.nvidia.com/cuda/cuda-c-programming-guide/

https://docs.nvidia.com/cuda-libraries/index.html

https://docs.nvidia.com/cuda/npp/index.html

https://docs.nvidia.com/cuda/npp/signal_filtering_functions.html

https://docs.nvidia.com/cuda/cublas/

Setting up Visual Studio for debugging: https://docs.nvidia.com/nsight-visual-studio-code-edition/cuda-debugger/index.html

Preview .md file side-by-side in Visual Studio: (Ctrl+K V)

