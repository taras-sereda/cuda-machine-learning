# GPU Programming

## Build

1. To build cuda examples, run: `make`. Compiled binaries will be saved in `bin` directory. To clean up, run `make clean`.
2. For triton kernels python vitual env should be created.

```bash
uv venv --python=python3.13
source .venv/bin/activate
pip install -r requirements.txt
```

To recompile python requiremenets: `uv pip compile requirements.in -o requirements.txt`

## Run samples

1. CUDA
   - Device info: `./bin/device_info`
   - Dynamic parallelism: `./bin/dynamic_parallelism`
2. Triton
   - Vector addition: `python src/triton_kernels/vector_add.py`

## General purpose GPU programming and related resources

1. [ZLUDA](https://github.com/vosen/ZLUDA) CUDA-like programming model for AMD GPUs, written in Rust.
1. [Hemi](https://github.com/harrism/hemi) (No longer maintened) library for writting reusable CPU and GPU code. Single kernel function executable on both device types. More in this [blog post](https://developer.nvidia.com/blog/simple-portable-parallel-c-hemi-2/).
1. [CUDA device management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html) functions refference guide.
1. [Holistic Trace Analysis](https://hta.readthedocs.io/en/latest/#) performance analysis of CUDA kernels. Allows to collect metrics on kernel execution time for distributed training systems.
1. [Blog on optimizations for matrix multiplications on GPUs](https://siboehm.com/articles/22/CUDA-MMM)
1. [CuBLAS docs](https://docs.nvidia.com/cuda/pdf/CUBLAS_Library.pdf)

### CUDA notes

1. Compile pytorch CUDA extension with multiple Compute Capabilities as well as PTX. <https://pytorch.org/docs/stable/cpp_extension.html>
`env TORCH_CUDA_ARCH_LIST="6.1 7.5 8.6+PTX" python setup.py install`

#### Quering device properties

`cudaDeviceGetAttribute` argued being [faster](https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/]) , though it's  too verbose

```cuda
  int maxBlockDimX;
  int maxBlockDimY;
  int maxBlockDimZ;
  cudaDeviceGetAttribute(&maxBlockDimX, cudaDevAttrMaxBlockDimX, devIdx);
  cudaDeviceGetAttribute(&maxBlockDimY, cudaDevAttrMaxBlockDimY, devIdx);
  cudaDeviceGetAttribute(&maxBlockDimZ, cudaDevAttrMaxBlockDimZ, devIdx);
  printf("  %d %d %d \n", maxBlockDimX, maxBlockDimY, maxBlockDimZ);
```

vs. `cudaGetDeviceProperties`

```cuda
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devIdx);
  printf("  Max Threads Dim: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
```

1. Compile CUDA kernels

```bash
nvcc device_info.cu -o device_info
```

2. Theory
Thread Hierarchy with easy examples.

```text
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
