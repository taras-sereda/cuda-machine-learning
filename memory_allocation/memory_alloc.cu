#include <stdio.h>
#include <unistd.h>

// Memory transfer host -> device -> host is essentially a 3 stage process:
// 1. Device memory allocation. Memory reservation
// 2. Host to device memory transfer.
// 3. Device memory de-allocation.

__device__ __constant__ float PI = 3.14159;
__device__ __shared__ float E;

__global__ void memory_access()
{
    int idx = threadIdx.x;

    printf("thread_idx=%d\n", idx);
    printf("reading PI value on CUDA PI=%f (shared between all kernels)\n", PI);
    if (idx == 5)
    {
        E = 2.71828; // each thread writes to shared memory;
    }
    // this is peculiar!
    // Threads will all run to this point, and once E will be assigned by thread_idx = 5,
    // value of E will be rewritten in global memory for all threads to E=2.71828
    // and corresponding value will be printed by each thread.
    __syncthreads();

    printf("reading E value on CUDA E=%f (shared between all threads in block)\n", E);
}
int main()

{
    int n_devices;
    cudaError_t err = cudaGetDeviceCount(&n_devices);
    printf("Error %s\n", cudaGetErrorString(err));
    printf("N devices %d\n", n_devices);

    // set CUDA device for all subsequent host-thread operations.
    int cur_device = 5;
    cudaSetDevice(cur_device);

    memory_access<<<1, 10>>>();

    float *A_d, *B_d, *C_d;
    size_t size = 5000000 * sizeof(float);
    cudaError_t alloc_err_a = cudaMalloc(&A_d, size);
    cudaError_t alloc_err_b = cudaMalloc(&B_d, size);
    cudaError_t alloc_err_c = cudaMalloc(&C_d, size);
    if (alloc_err_a != cudaSuccess)
    {
        printf("Allocation Error A_d %s\n", cudaGetErrorString(alloc_err_a));
    }
    if (alloc_err_b != cudaSuccess)
    {
        printf("Allocation Error B_d %s\n", cudaGetErrorString(alloc_err_b));
    }
    if (alloc_err_c != cudaSuccess)
    {
        printf("Allocation Error C_d %s\n", cudaGetErrorString(alloc_err_c));
    }
    sleep(10);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    // an attempt of double free!
    cudaFree(&A_d);
}