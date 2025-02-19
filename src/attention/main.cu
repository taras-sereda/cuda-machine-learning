#include "../kernel.cu"
#include "../util.c"


#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>


int main() {
    int a_rows = 4;
    int a_cols = 8;
    int b_rows = 8;
    int b_cols = 2;

    // static_assert(a_cols == b_rows, "Matrix dimensions are not compatible for multiplication");
    int a_size = a_rows*a_cols * sizeof(float);
    int b_size = b_rows*b_cols * sizeof(float);
    int c_size = a_rows*b_cols * sizeof(float);

    float *A_host = (float *)malloc(a_size);
    float *B_host = (float *)malloc(b_size);
    float *C_host = (float *)malloc(c_size);

    // init host data
    init(A_host, a_rows*a_cols);
    init(B_host, b_rows*b_cols);

    printf("A:\n");
    print_tensor(A_host, a_rows*a_cols);
    printf("\n");

    printf("B:\n");
    print_tensor(B_host, b_rows*b_cols);
    printf("\n");    

    float *A_device;
    float *B_device;
    float *C_device;

    cudaMalloc(&A_device, a_size);
    cudaMalloc(&B_device,  b_size);
    cudaMalloc(&C_device, c_size);

    cudaMemcpy(A_device, A_host, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, b_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);

    dim3 numBlocks((b_rows + threadsPerBlock.x - 1) / threadsPerBlock.x, (a_cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("%d %d \n", numBlocks.x, numBlocks.y);

    matMulKernel<<<numBlocks, threadsPerBlock>>>(A_device, B_device, C_device, a_rows, a_cols, b_cols);

    cudaMemcpy(C_host, C_device, c_size, cudaMemcpyDeviceToHost);

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    print_tensor(C_host, a_rows*b_cols);
    free(A_host);
    free(B_host);
    free(C_host);
    
    return 0;
}