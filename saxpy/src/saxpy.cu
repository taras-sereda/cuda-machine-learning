#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, float * restrict x, float * b) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a * x[i] + y[i];
} 


int main(int argc, char** argv) {
	
	int N = 1<<20; // same as 2^20 or around 1 million. 	
	cudaMemcpy(d_x, x, N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N, cudaMemcpyHostToDevice);

	saxpy<<4096, 256>>(N, 2.0, d_x, d_y);

	cudaMemcpy(y, d_y, N, cudaMemcpyDeviceToHost);
}
