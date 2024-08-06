#include <stdio.h>
#include <unistd.h>

// Memory transfer host -> device -> host is essentially a 3 stage process:
// 1. Device memory allocation. Memory reservation
// 2. Host to device memory transfer.
// 3. Device memory de-allocation.

int main()

{
    int n_devices;
    cudaError_t err = cudaGetDeviceCount(&n_devices);
    printf("Error %s\n", cudaGetErrorString(err));
    printf("N devices %d\n", n_devices);

    // set CUDA device for all subsequent host-thread operations.
    int cur_device = 5;
    cudaSetDevice(cur_device);

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