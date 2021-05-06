#include <iostream>
#include <math.h>

__global__
void add_single_threaded(int n, float *x, float *y){
    for (int i = 0; i < n; i ++)
        y[i] = x[i] + y[i];
}

__global__
void add_multi_threaded(int n, float *x, float *y){

    int idx = threadIdx.x;
    int stride = blockDim.x;
    // printf("threadIdx = %d; blockDim = %d; blockIdx = %d %d\n", idx, stride, blockIdx.x, blockIdx.y);
    for (int i = idx; i < n; i += stride)
    {
        printf("array idx %d\n", i);
        y[i] = x[i] + y[i];
    }
}

__global__
void add(int n, float *x, float *y)
{   
    // gird-stride loop.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // printf("threadIdx = %d; blockDim = %d; blockIdx = %d\n", idx, stride, blockIdx.x);
    for (int i = idx; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20; // ~1M elements, more preciesly 2^20 = 1048576 elements.:)
    float *x, *y;

    // Unified memory allocation, accessible by CPUs and GPUs
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // init x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }


    // Run kernel on 1M elements on the GPU
    // add_single_threaded<<<1, 1>>>(N, x, y);


    // Run kernel on a single block of multiple threads
    int blockSize = 256;
    // add_multi_threaded<<<1, blockSize>>>(N, x, y);

    // Run kernel on multiple thread blocks
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
