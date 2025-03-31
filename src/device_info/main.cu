#include <iostream>
#include <stdio.h>

#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_COLOR_CYAN "\x1B[36m"

int main()
{
  int nDevices;

  cudaError_t err = cudaGetDeviceCount(&nDevices);
  if (err != cudaSuccess)
    printf("%s\n", cudaGetErrorString(err));

  printf("\nCUDA Runtime version: %d.%d\n\n", CUDART_VERSION / 1000, (CUDART_VERSION % 100) / 10); // CUDART_VERSION is defined in cuda_runtime_api.h

  for (int devIdx = 0; devIdx < nDevices; devIdx++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devIdx);
    printf("Device Number: %d\n", devIdx);
    printf("  Device name: %s\n", prop.name);
    printf("  Device compute capability: %d.%d\n", prop.major, prop.minor);

    // Print the UUID
    printf("  UUID: ");
    for (int i = 0; i < 16; ++i)
    {
      printf("%02x", (unsigned char)prop.uuid.bytes[i]);
      if (i % 4 == 3 && i != 15)
        printf("-");
    }
    printf("\n");

    printf("  LUID: %s(undefined on non-Windows)\n", prop.luid);
    printf("  CanMapHostMemory: %d\n", prop.canMapHostMemory); // True if device supports zero-copy memory mapping. When cuda threads can access host memory directly
    printf("  Clock rate (KHz): %d\n", prop.clockRate);
    printf("  Memory clock rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory bus width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak memory bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    printf("  Multiprocessor count, aka SMs: %d\n", prop.multiProcessorCount);
    printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Registers per block: %d (can be less then per SM!)\n", prop.regsPerBlock);
    printf("  Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("  Warp Size: %d\n", prop.warpSize);

    printf("  Max grid size: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  %sMax threads dim: %d %d %d%s\n", ANSI_COLOR_CYAN, prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], ANSI_COLOR_RESET);
  }
}
