compilation on arch linux

### Requirements:

yay -S glu freeglut

### Compilation:

CPU version
`g++ julia_set.cpp -o julia_set -lGLU -lGL -lglut`

GPU version

`nvcc julia_set.cu -o julia_set_gpu -lGLU -lGL -lglut`


### Notes:

`gridDim.x` - is a tuple of 3 elements. But CUDA supported only 2d grids for a long time, last dim was just singular. Need to check if CUDA supports now 3d grids.

`__device__` function specifier. Tells compiler that function is callable from device code and runs on device. CUDA - CUDA.

where

`__global__` is CPU - CUDA.



### Questions:
1. How can I compile same C++ and CUDA versions with `clang`?
