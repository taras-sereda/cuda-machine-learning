#### Quering device properties.

`cudaDeviceGetAttribute   `argued being [faster](https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/]) , though it's  too verbose

```
  int maxBlockDimX;
  int maxBlockDimY;
  int maxBlockDimZ;
  cudaDeviceGetAttribute(&maxBlockDimX, cudaDevAttrMaxBlockDimX, devIdx);
  cudaDeviceGetAttribute(&maxBlockDimY, cudaDevAttrMaxBlockDimY, devIdx);
  cudaDeviceGetAttribute(&maxBlockDimZ, cudaDevAttrMaxBlockDimZ, devIdx);
  printf("  %d %d %d \n", maxBlockDimX, maxBlockDimY, maxBlockDimZ);
```

vs. `cudaGetDeviceProperties`

```
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devIdx);
  printf("  Max Threads Dim: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
```



- [reference of CUDA device properties](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)

- [Hemi](https://github.com/harrism/hemi) - library for writting reusable CPU and GPU code. single kernel function executable on both device types. More in this [blog post](https://developer.nvidia.com/blog/simple-portable-parallel-c-hemi-2/).



1. Compile CUDA kernels
```
nvcc device_info.cu -o device_info
```

2. Theory
Thread Hierarchy with easy examples.
```
1d(Dx=15)
like finiding a point on a line
threadIdx = (7)
threadId = threadIdx.x
threadId = 7

-> -------x-------

2d(Dx=15, Dy=4)
like finding a line on a plane and then finding a point on it.
threadIdx = (2, 7)
threadId = threadIdx.x + threadIdx.y * Dx
threadId = 7 + 2 * 15 = 37
   ---------------
   ---------------
-> -------x-------
   ---------------



3d(Dx=15, Dy=4, Dz=2)
threadIdx = (0, 2, 7)
threadId = threadIdx.x + threadIdx.y * Dx + threadIdx.z * Dx * Dy
like finding a plane in a volume then line on a plane and finaly a point.

   --------------- --
   ---------------   | first
-> --------x------   | find a slice
   --------------- --

   ---------------
   --------------- 
   --------------- 
   --------------- 
```