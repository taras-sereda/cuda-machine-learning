#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#define BLOCK_SIZE 32

typedef struct
{
    int width;
    int height;
    float *elements;
} Matrix;

__global__ void matTransposeKernelNaive(Matrix A, Matrix B)
{

    int row_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row_idx < A.height && col_idx < A.width)
    {
        B.elements[B.width * col_idx + row_idx] =
            A.elements[A.width * row_idx + col_idx];
    }
}

__global__ void matTransposeKernelStrided(Matrix A, Matrix B)
{

    int row_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    // Bank conflict resolvement magic.
    //__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE+1];

    if (row_idx < A.height && col_idx < A.width)
    {
        tile[threadIdx.y][threadIdx.x] = A.elements[A.width * row_idx + col_idx];
    }
    __syncthreads();

    // Transposed offsets.
    int trans_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int trans_col = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // printf("Thread (%d, %d) in block (%d, %d)\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

    if (trans_row < B.height && trans_col < B.width)
    {
        // Coalesced write to a transposed location.
        B.elements[B.width * trans_row + trans_col] =
            tile[threadIdx.y][threadIdx.x];
    }
}

__global__ void matTransposeKernel(Matrix A, Matrix B)
{

    int row_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    // Bank conflict resolvement magic.
    //__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE+1];

    if (row_idx < A.height && col_idx < A.width)
    {
        tile[threadIdx.y][threadIdx.x] = A.elements[A.width * row_idx + col_idx];
    }
    __syncthreads();

    // Transposed offsets.
    int trans_row = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    int trans_col = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    // printf("Thread (%d, %d) in block (%d, %d)\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

    if (trans_row < B.height && trans_col < B.width)
    {
        // Coalesced write to a transposed location.
        B.elements[B.width * trans_row + trans_col] =
            tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void matTransposeKernelOptim(Matrix A, Matrix B)
{

    int row_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Bank conflict resolvement magic.
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE+1];

    if (row_idx < A.height && col_idx < A.width)
    {
        tile[threadIdx.y][threadIdx.x] = A.elements[A.width * row_idx + col_idx];
    }
    __syncthreads();

    // Transposed offsets.
    int trans_row = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    int trans_col = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    // printf("Thread (%d, %d) in block (%d, %d)\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

    if (trans_row < B.height && trans_col < B.width)
    {
        // Coalesced write to a transposed location.
        B.elements[B.width * trans_row + trans_col] =
            tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void matTransposeKernel1DShmem(Matrix A, Matrix B)
{

    int row_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Bank conflict resolvement magic.
    __shared__ float tile[BLOCK_SIZE*(BLOCK_SIZE+1)];
    uint stride = BLOCK_SIZE + 1;

    if (row_idx < A.height && col_idx < A.width)
    {
        tile[threadIdx.y * stride + threadIdx.x] = A.elements[A.width * row_idx + col_idx];
    }
    __syncthreads();

    // Transposed offsets.
    int trans_row = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    int trans_col = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    // printf("Thread (%d, %d) in block (%d, %d)\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

    if (trans_row < B.height && trans_col < B.width)
    {
        // Coalesced write to a transposed location.
        B.elements[B.width * trans_row + trans_col] =
            tile[threadIdx.x * stride + threadIdx.y];
    }
}
void init_matrix(float *elements, int width, int height)
{

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int idx = width * i + j;
            elements[idx] = rand() / (float)RAND_MAX;
        }
    }
}

void check_res(Matrix A, Matrix B)
{

    for (int i = 0; i < A.height; i++)
    {
        for (int j = 0; j < A.width; j++)
        {
            int a_idx = A.width * i + j;
            int b_idx = A.height * j + i;
            printf("Matrix A [%d][%d] = %.4f Matrix B [%d][%d] = %.4f\n",
                   i, j, A.elements[a_idx], j, i, B.elements[b_idx]);
        }
    }
}

int main(int argc, char **argv)
{
    int height = 1024*8+7;
    int width = 2048*8+1;
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

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float time_ms;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((h_A.width + BLOCK_SIZE - 1) / BLOCK_SIZE, (h_A.height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // warmup
    matTransposeKernelNaive<<<dimGrid, dimBlock>>>(d_A, d_B);
    cudaEventRecord(startEvent, 0);
    matTransposeKernelNaive<<<dimGrid, dimBlock>>>(d_A, d_B);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time_ms, startEvent, stopEvent);
    printf("Naive. time taken: %f, bandwidth: %f GB/s\n", time_ms, width * height * 2 * sizeof(float) * 1e-6 / time_ms);

    // warmup
    matTransposeKernelStrided<<<dimGrid, dimBlock>>>(d_A, d_B);
    cudaEventRecord(startEvent, 0);
    matTransposeKernelStrided<<<dimGrid, dimBlock>>>(d_A, d_B);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time_ms, startEvent, stopEvent);
    printf("Strided. time taken: %f, bandwidth: %f GB/s\n", time_ms, width * height * 2 * sizeof(float) * 1e-6 / time_ms);

    // warmup
    matTransposeKernel<<<dimGrid, dimBlock>>>(d_A, d_B);
    cudaEventRecord(startEvent, 0);
    matTransposeKernel<<<dimGrid, dimBlock>>>(d_A, d_B);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time_ms, startEvent, stopEvent);
    printf("Coalesced. time taken: %f, bandwidth: %f GB/s\n", time_ms, width * height * 2 * sizeof(float) * 1e-6 / time_ms);

    // warmup
    matTransposeKernelOptim<<<dimGrid, dimBlock>>>(d_A, d_B);
    cudaEventRecord(startEvent, 0);
    matTransposeKernelOptim<<<dimGrid, dimBlock>>>(d_A, d_B);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time_ms, startEvent, stopEvent);
    printf("Optim[bank conflicts prevention] time taken: %f, bandwidth: %f GB/s\n", time_ms, width * height * 2 * sizeof(float) * 1e-6 / time_ms);

    // warmup
    matTransposeKernel1DShmem<<<dimGrid, dimBlock>>>(d_A, d_B);
    cudaEventRecord(startEvent, 0);
    matTransposeKernel1DShmem<<<dimGrid, dimBlock>>>(d_A, d_B);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&time_ms, startEvent, stopEvent);
    printf("1D shem time taken: %f, bandwidth: %f GB/s\n", time_ms, width * height * 2 * sizeof(float) * 1e-6 / time_ms);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_B.elements, d_B.elements, B_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);

    //check_res(h_A, h_B);

    free(h_A.elements);
    free(h_B.elements);
}
