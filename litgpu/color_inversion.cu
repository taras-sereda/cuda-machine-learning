#include <cuda_runtime.h>
#include <stdio.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {

    int stride = 4;
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * stride;
    if (idx >= width * height * stride) {
        return;
    }
    image[idx] = 255 - image[idx];
    image[idx+1] = 255 - image[idx+1];
    image[idx+2] = 255 - image[idx+2];
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
