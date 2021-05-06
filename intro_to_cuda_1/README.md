notes and code samples from https://developer.nvidia.com/blog/even-easier-introduction-cuda/

compilation:
   - CPU code `clang++ add.cpp -o add`
   - GPU code `nvcc add.cu -o add`

`__global__` - function specifier informs `nvcc` that function runs on CUDA and can be called from CPU code. Additionally this specifier is not recognizable by `clang++` and causes compilation error. 

##### Profiling

`nvprof ./add_cuda` - launch command line GPU profiler to get stats on kernel execution.

- profiling of single thread kernal launch

```
==24959== NVPROF is profiling process 24959, command: ./add_cuda
Max error: 0
==24959== Profiling application: ./add_cuda
==24959== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  263.83ms         1  263.83ms  263.83ms  263.83ms  add(int, float*, float*)
      API calls:   65.95%  263.84ms         1  263.84ms  263.84ms  263.84ms  cudaDeviceSynchronize
                   33.24%  132.97ms         2  66.484ms  581.60us  132.39ms  cudaMallocManaged
                    0.37%  1.4920ms         1  1.4920ms  1.4920ms  1.4920ms  cudaLaunchKernel
                    0.16%  637.12us       192  3.3180us     100ns  192.76us  cuDeviceGetAttribute
                    0.16%  624.71us         2  312.35us  284.71us  340.00us  cudaFree
                    0.11%  436.70us         2  218.35us  217.91us  218.79us  cuDeviceTotalMem
                    0.01%  40.861us         2  20.430us  17.549us  23.312us  cuDeviceGetName
                    0.00%  4.4190us         2  2.2090us  1.0550us  3.3640us  cuDeviceGetPCIBusId
                    0.00%     796ns         3     265ns      95ns     543ns  cuDeviceGetCount
                    0.00%     737ns         4     184ns      94ns     398ns  cuDeviceGet
                    0.00%     309ns         2     154ns     140ns     169ns  cuDeviceGetUuid

==24959== Unified Memory profiling result:
Device "GeForce GTX TITAN X (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       6  1.3333MB  768.00KB  2.0000MB  8.000000MB  1.310752ms  Host To Device
     102  120.47KB  4.0000KB  0.9961MB  12.00000MB  1.986528ms  Device To Host
Total CPU Page faults: 51
```

- profiling of multithread kernel launch