notes and code samples from https://developer.nvidia.com/blog/even-easier-introduction-cuda/



### Compilation

   - CPU code `clang++ add.cpp -o add`
   - GPU code `nvcc add.cu -o add`



### Notes

`__global__` - function specifier informs `nvcc` that function runs on CUDA and can be called from CPU code. Additionally this specifier is not recognizable by `clang++` and causes compilation error. 

`<<<numBlocks, blockSize>>>` - execution configuration of CUDA Kernel.

printf insdide kernel significantly slows down kernel execution time.

gridDim represents - number of thread-blocks. **grid** - is a collection of blocks of parallel threads. In case of 1-d indexing `gridDim.x * blockDim.x` is a total number of threads in the grid.

Threads are running asynchornously within a block.

**Grid-stride style kernel** is a generic approach to launch a kernel even if number of threads is less then number of parallel computations necessary to accomplish. More on this [topic](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)

```c++
// Blocks of threads are mulitple of 32.
blockSize = 64
numBlocks = 32

<<<numBlocks, blockSize>>>add(N, x, y);
```



### Profling

`nvprof ./add_cuda` - launch command line GPU profiler to get stats on kernel execution.

```

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  263.83ms         1  263.83ms  263.83ms  263.83ms  add_single_threaded(int, float*, float*)
 GPU activities:  100.00%  2.0172ms         1  2.0172ms  2.0172ms  2.0172ms  add_multi_threaded(int, float*, float*)
 GPU activities:  100.00%  51.137us         1  51.137us  51.137us  51.137us  add(int, float*, float*)
```

