

1. Dynamic parallelism enabled cuda programs can be [compiled in multiple ways](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-options-for-separate-compilation):

- `nvcc --device-c -o main.o main.cu && nvcc -o main main.cu`
- `nvcc  -rdc=true main.cu -o main -lcudadevrt`, note that `-lcudadevrt` is optional for modern CUDA versions. Where deviceruntime is already linked by default.