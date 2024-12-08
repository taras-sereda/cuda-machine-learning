#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <stdlib.h>
#include <libtar.h>
#include <fcntl.h>
#include <string.h>

#include "io.h"

// Utility function to check CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void benchmarkD2HTransfer(size_t size) {
    // Allocate host memory
    float* h_data;
    CHECK_CUDA(cudaMallocHost(&h_data, size)); // Using pinned memory for better performance

    // Allocate device memory
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));

    // Initialize device memory (optional)
    CHECK_CUDA(cudaMemset(d_data, 0, size));

    // Warm up transfer
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Number of iterations for averaging
    const int iterations = 100;
    
    // Synchronize before timing
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark transfer
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate statistics
    double elapsed = std::chrono::duration<double>(end - start).count();
    double avgTime = elapsed / iterations;
    double bandwidth = (size / (1024.0 * 1024.0 * 1024.0)) / avgTime; // GB/s

    printf("Transfer size: %.2f MB\n", size / (1024.0 * 1024.0));
    printf("Average transfer time: %.3f ms\n", avgTime * 1000);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);

    // Cleanup
    CHECK_CUDA(cudaFreeHost(h_data));
    CHECK_CUDA(cudaFree(d_data));
}

// int main() {
//     // Print device info
//     cudaDeviceProp prop;
//     CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
//     printf("Device: %s\n", prop.name);
//     printf("Max Memory Transfer Rate: %.2f GB/s\n\n", 
//            prop.memoryBusWidth * prop.memoryClockRate * 2.0 / (8.0 * 1000.0 * 1000.0));

//     // Test different transfer sizes
//     const size_t sizes[] = {
//         1024 * 1024,        // 1 MB
//         16 * 1024 * 1024,   // 16 MB
//         256 * 1024 * 1024,  // 256 MB
//         1024 * 1024 * 1024  // 1 GB
//     };

//     for (size_t size : sizes) {
//         printf("\n=== Testing transfer size: %.2f MB ===\n", size / (1024.0 * 1024.0));
//         benchmarkD2HTransfer(size);
//     }

//     return 0;
// }





int main(int argc, char *argv[]) {
    TAR *tar;
    tartype_t *type = NULL;
    char *tarfile;
    int ret;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <tarfile>\n", argv[0]);
        return 1;
    }
    
    tarfile = argv[1];
    
    // Open the tar file
    ret = tar_open(&tar, tarfile, type, O_RDONLY, 0, TAR_VERBOSE);
    if (ret != 0) {
        fprintf(stderr, "Error opening tar file %s\n", tarfile);
        return 1;
    }

    // Read through all entries
    while ((ret = th_read(tar)) == 0) {
        char *filename = th_get_pathname(tar);
        unsigned int size = th_get_size(tar);
        unsigned int mode = th_get_mode(tar);
        unsigned int mtime = th_get_mtime(tar);
        
        
        if (filename == NULL || size == 0) {
            continue;
        }

        print_file_info(filename, size, mode, mtime);

        // If it's a regular file, you can read its contents
        if (TH_ISREG(tar)) {
            char buffer[1024];
            size_t bytes_read;
            
            // Skip file contents - remove this if you want to process the contents
            tar_skip_regfile(tar);
            
            // /* To read file contents, uncomment this block:
            while ((bytes_read = tar_block_read(tar, buffer) > 0)) {
                // Process the file contents here
                // For example, to print contents:
                fwrite(buffer, 1, bytes_read, stdout);
            }
            // */
        }
    }

    // Check if we reached end of archive
    if (ret != 1) {
        fprintf(stderr, "Error while reading tar file\n");
    }

    // Close the tar file
    tar_close(tar);

    return 0;
}
