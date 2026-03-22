#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#define BLOCK_SIZE 16 

typedef struct {
    int width;
    int height;
    float *elements;
} Matrix;

__global__ void matTransposeKernel(Matrix A, Matrix B) {

    int row_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;


    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];


    if (row_idx < A.height && col_idx < A.width) {
        tile[threadIdx.y][threadIdx.x] = A.elements[A.width * row_idx + col_idx];

    }
    __syncthreads();

    // Why?
    int trans_row = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    int trans_col = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (trans_row < B.height && trans_col < B.width) {
        B.elements[B.width * trans_row + trans_col] =
           tile[threadIdx.x][threadIdx.y];
    }
}
 
void init_matrix(float *elements, int width, int height) {

    for(int i=0; i< height; i++){
        for(int j=0; j< width; j++){
            int idx = width * i + j;
            elements[idx] = random()/(float)RAND_MAX;
        }
    }
}


void check_res(Matrix A, Matrix B) {

    for(int i=0; i<A.height; i++){
        for(int j=0; j<A.width; j++){
            int a_idx = A.width * i + j;
            int b_idx = A.height * j + i;
            printf("Matrix A [%d][%d] = %.4f\n", i, j, A.elements[a_idx]);
            printf("Matrix B [%d][%d] = %.4f\n", j, i, B.elements[b_idx]);
        }
    }
}

int main(int argc, char** argv) {
    int height = 1024;
    int width = 2049;
    Matrix h_A;
    h_A.height = height;
    h_A.width = width;
    size_t A_size = h_A.height * h_A.width * sizeof(float);
    h_A.elements = (float *)malloc(A_size);
    init_matrix(h_A.elements, h_A.width, h_A.height);

    Matrix d_A;
    d_A.height = h_A.height;
    d_A.width = h_A.width;
    cudaMalloc(&d_A.elements, A_size);


    Matrix h_B;
    h_B.height = width;
    h_B.width = height;
    size_t B_size = h_B.height * h_B.width * sizeof(float);
    h_B.elements = (float *)malloc(B_size);

    Matrix d_B;
    d_B.height = h_B.height;
    d_B.width = h_B.width;
    cudaMalloc(&d_B.elements, B_size);

    cudaMemcpy(d_A.elements, h_A.elements, A_size, cudaMemcpyHostToDevice);


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((h_A.width + BLOCK_SIZE -1)/BLOCK_SIZE, (h_A.height + BLOCK_SIZE -1)/BLOCK_SIZE);

    matTransposeKernel<<<dimGrid, dimBlock>>>(d_A, d_B);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_B.elements, d_B.elements, B_size, cudaMemcpyDeviceToHost);

    check_res(h_A, h_B);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);

    free(h_A.elements);
    free(h_B.elements);

}


