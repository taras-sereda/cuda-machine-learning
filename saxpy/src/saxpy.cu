#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, float* x, float* y) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = a * x[i] + y[i];
} 


int main(int argc, char** argv) {

        int nDevices; 
        cudaGetDeviceCount(&nDevices);  

        printf("Number of GPU devices %d\n", nDevices);

        for (int i = 0; i < nDevices; i++) {
       
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
           
            printf("Device Number: %d\n", i); 
            printf("  Device Name: %s\n", prop.name);
            printf("  Memory Clock Rate (Khz): %d\n", prop.memoryClockRate);
            printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
            printf("  Peak Memory Bandwidth (GB/s): %f\n",
                    2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        }	

	int N = 1<<20; // same as 2^20 or around 1 million. 	
        float *x, *y, *d_x, *d_y;
        x = (float*)malloc(N*sizeof(float));
        y = (float*)malloc(N*sizeof(float));
        
        cudaMalloc(&d_x, N*sizeof(float));
        cudaMalloc(&d_y, N*sizeof(float));
       
        printf("size of float %lu\n", sizeof(float)); 
        sleep(3);
        printf("sleeping a bit.\n"); 

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

	saxpy<<<4096, 256>>>(N, 2.0, d_x, d_y);

	cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
}
