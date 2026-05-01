#include <stdio.h>

__global__ void atomic_add(float *acc, int n) {
  int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (th_idx < n) {
    atomicAdd(&acc[0], 1.0f / (th_idx + 1));
  }
}

int main() {
  int n = 128;

  float h_data[n];
  for (int i = 0; i < n; i++)
    h_data[i] = 0.0f;

  float *d_data;
  cudaMalloc(&d_data, sizeof(float) * n);
  cudaMemset(d_data, 0, sizeof(float) * n);
  int n_th = 256;
  int n_blck = (n + n_th - 1) / n_th;

  atomic_add<<<n_blck, n_th>>>(d_data, n);

  cudaMemcpy(h_data, d_data, sizeof(float) * n, cudaMemcpyDeviceToHost);
  printf("acc[0] = %f\n", h_data[0]);

  cudaFree(d_data);
  return 0;
}
