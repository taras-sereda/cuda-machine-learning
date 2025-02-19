#include "../kernel.cu"
#include "../util.c"

#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  int a_rows = 4;
  int a_cols = 8;
  int b_rows = 8;
  int b_cols = 2;

  float *A_host, *B_host, *C_host;
  float *A_device, *B_device, *C_device;

  int a_size = a_rows * a_cols * sizeof(float);
  int b_size = b_rows * b_cols * sizeof(float);
  int c_size = a_rows * b_cols * sizeof(float);

  A_host = (float *)malloc(a_size);
  B_host = (float *)malloc(b_size);
  C_host = (float *)malloc(c_size);

  // init host data
  init(A_host, a_rows * a_cols);
  init(B_host, b_rows * b_cols);

  cudaMalloc((void **)&A_device, a_size);
  cudaMalloc((void **)&B_device, b_size);
  cudaMalloc((void **)&C_device, c_size);

  cudaMemcpy(A_device, A_host, a_size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_device, B_host, b_size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);

  dim3 numBlocks((b_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (a_cols + threadsPerBlock.y - 1) / threadsPerBlock.y);

  naiveMatMulKernel<<<numBlocks, threadsPerBlock>>>(
      A_device, B_device, C_device, a_rows, a_cols, b_cols);

  cudaMemcpy(C_host, C_device, c_size, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cudaFree(A_device);
  cudaFree(B_device);
  cudaFree(C_device);

  print_tensor(C_host, a_rows * b_cols);

  free(A_host);
  free(B_host);
  free(C_host);

  return 0;
}