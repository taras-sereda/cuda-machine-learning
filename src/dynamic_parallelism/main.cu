#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void tail_launch(int *data) {
    data[threadIdx.x] = data[threadIdx.x] + 1;
}

__global__ void child_launch(int *data) {
    data[threadIdx.x] = data[threadIdx.x] + 1;
}

__global__ void parent_launch(int *data) {
    data[threadIdx.x] = threadIdx.x;

    __syncthreads();

    if (threadIdx.x == 0) {
        child_launch<<<1, 256>>>(data);
        // Adds implict synchronization point to prevent parrent thread from exiting,
        // allowing access to the results of previous kernel launch.
        tail_launch<<<1, 256, 0, cudaStreamTailLaunch>>>(data);
    }

}

int main() {
    int n_elem = 256;

    // allocate host memory
    int * data;
    data = (int*)malloc(sizeof(int) * n_elem);

    // allocate device memory
    int * device_data;
    cudaMalloc(&device_data, sizeof(int) * n_elem);

    parent_launch<<<1, 256>>>(device_data);

    cudaMemcpy(data, device_data, sizeof(int) * n_elem, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < n_elem; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    cudaFree(device_data);
    free(data);
    return 0;
}