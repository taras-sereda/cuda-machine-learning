notes and code samples from https://developer.nvidia.com/blog/even-easier-introduction-cuda/

compilation:
   - CPU code `clang++ add.cpp -o add`
   - GPU code `nvcc add.cu -o add`

`__global__` - function specifier informs `nvcc` that function runs on CUDA and can be called from CPU code. Additionally this specifier is not recognizable by `clang++` and causes compilation error. 

