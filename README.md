# Cuda At Scal Indemendent Project

For this project I decided to implement interpolation using CUDA.  Why interpolation?  Interpolation is useful in its own right.  At the same time once interpolation has been implemented it is not to difficult to do decimation, correlation, and filtering.



## Commands to make the project:

```
git submodule update --init --recursive
cd src
make
```
## Commands to run the executable:
```
./newProjectTest.exe 
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

Setting up Visual Studio for debugging: https://docs.nvidia.com/nsight-visual-studio-code-edition/cuda-debugger/index.html

Preview .md file side-by-side in Visual Studio: (Ctrl+K V)

