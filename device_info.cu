#include <iostream>
#include <stdio.h> 


#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_COLOR_CYAN  "\x1B[36m"

int main() {  
  int nDevices;

  cudaError_t err = cudaGetDeviceCount(&nDevices);
  printf("%s\n", cudaGetErrorString(err));
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

  for (int devIdx = 0; devIdx < nDevices; devIdx++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devIdx);
    printf("Device Number: %d\n", devIdx);
    printf("  Device name: %s\n", prop.name);
    printf("  Device compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  LUID: %s\n", prop.luid);
    printf("  CanMapHostMemory: %d\n", prop.canMapHostMemory);
    printf("  Clock Rate (KHz): %d\n", prop.clockRate);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

    printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  %sMax Threads Dim: %d %d %d%s\n", ANSI_COLOR_CYAN, prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], ANSI_COLOR_RESET);
    printf("  Max Grid Size: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);

  }
}
