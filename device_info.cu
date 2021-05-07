#include <iostream>
#include <stdio.h> 


#define ANSI_COLOR_RESET "\x1b[0m"
#define ANSI_COLOR_CYAN  "\x1B[36m"

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int devIdx = 0; devIdx < nDevices; devIdx++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devIdx);
    printf("Device Number: %d\n", devIdx);
    printf("  Device name: %s\n", prop.name);
    printf("  Device compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  LUID: %s\n", prop.luid);
    printf("  CanMapHostMemory: %d\n", prop.canMapHostMemory);
    printf("  Clock Rate (KHz): %d\n",
           prop.clockRate);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

    // https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/
    printf("  %s%s%s \n", ANSI_COLOR_CYAN, "Querying CUDA Attributes via cudaDeviceGetAttribute", ANSI_COLOR_RESET);

    int memoryClockRate;
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, devIdx);
    printf("  Memory Clock Rate (KHz): %d\n", memoryClockRate);
    //int maxThreadsPerBlock;
    int maxThreadsDim [3];
    int maxGridSize [3];
    //cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, devIdx);
    //cudaDeviceGetAttribute(&maxThreadsDim, cudaDevAttrMaxThreadsDim, devIdx);
    //cudaDeviceGetAttribute(&maxGridSize, cudaDevAttrMaxGridSize, devIdx);
    printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    // printf("  Max Threads Dim: %d\n", &prop.maxThreadsDim[0]);
    // printf("  Max Grid Size: %d\n", &prop.maxGridSize[0]);
    std::cout << "  Max Thread Dim x,y,z " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "  Max Grid Size x,y,z " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
  }
}
