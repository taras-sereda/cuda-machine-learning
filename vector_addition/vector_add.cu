#include <stdio.h>
#include <time.h>

__device__ float add2(float elem)
{
    return elem + 2;
}
__global__ void vecAddKernel(float *A, float *B, float *C, int n_elem, bool debug)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n_elem)
    {
        C[idx] = add2(A[idx] + B[idx]);
    }
    if (debug)
    {
        printf("%d %f %f %f\n", idx, A[idx], B[idx], C[idx]);
    }
}

void init_vector(float *elems, int n_elem)
{

    for (int i = 0; i < n_elem; i++)
    {
        elems[i] = random() / (float)RAND_MAX;
    }
}

void print_vector(float *elems, int n_elem)
{
    for (int i = 0; i < n_elem; i++)
    {
        printf("%f ", elems[i]);
    }
    printf("\n");
}

void vector_add(float *A_h, float *B_h, float *C_h, int n_elem, bool debug)
{
    float *A_d, *B_d, *C_d;
    size_t size = n_elem * sizeof(float);

    // 1. Allocate memory on device
    cudaError_t alloc_err_a = cudaMalloc(&A_d, size);
    cudaError_t alloc_err_b = cudaMalloc(&B_d, size);
    cudaError_t alloc_err_c = cudaMalloc(&C_d, size);
    if (alloc_err_a != cudaSuccess)
    {
        printf("Allocation Error A_d %s\n", cudaGetErrorString(alloc_err_a));
    }
    if (alloc_err_b != cudaSuccess)
    {
        printf("Allocation Error B_d %s\n", cudaGetErrorString(alloc_err_b));
    }
    if (alloc_err_c != cudaSuccess)
    {
        printf("Allocation Error C_d %s\n", cudaGetErrorString(alloc_err_c));
    }

    // 2. Copy data from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // 3. Call CUDA kernel
    vecAddKernel<<<ceil(n_elem / 256.0), 256>>>(A_d, B_d, C_d, n_elem, debug);

    // 4. Copy data from device to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    if (debug)
    {
        print_vector(A_h, n_elem);
        print_vector(B_h, n_elem);
    }

    print_vector(C_h, n_elem);

    // 5. Dealocate memory on device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
    // Seed random number generator with the current time
    srand(time(NULL));

    int n_elem = 12;
    size_t size = n_elem * sizeof(float);
    float *A_h = (float *)malloc(size);
    float *B_h = (float *)malloc(size);
    float *C_h = (float *)malloc(size);
    bool debug = false;
    init_vector(A_h, n_elem);
    init_vector(B_h, n_elem);

    vector_add(A_h, B_h, C_h, n_elem, debug);
}