#include <stdio.h>

__global__ void atomic_add(float *acc, int n) {
  int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (th_idx < n) {
    atomicAdd(&acc[0], 1.0f / (th_idx + 1));
  }
}

int main() {
  int n_repeat = 10;
  int n = 1024;

  float results[n_repeat];

  for (int iter = 0; iter < n_repeat; iter++) {

    float *d_data;
    cudaMalloc(&d_data, sizeof(float) * n);
    cudaMemset(d_data, 0, sizeof(float) * n);
    int n_th = 256;
    int n_blck = (n + n_th - 1) / n_th;

    atomic_add<<<n_blck, n_th>>>(d_data, n);

    cudaMemcpy(&results[iter], d_data, sizeof(float), cudaMemcpyDeviceToHost);
    printf("iter %d: result = %.10f\n", iter, results[iter]);

    cudaFree(d_data);
  }


  for (int iter = 1; iter < n_repeat; iter++) {
    float drift = abs(results[iter-1] - results[iter]);
    printf("iter %d: drift wrt prev iter = %.10f\n", iter, drift);
  }

  return 0;
}
