/*
 ============================================================================
 Name        : matMul.cu
 Author      : taras-sereda
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */
#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#define BLOCK_SIZE 32

typedef struct {

	int width;
	int height;
	float *elements;
} Matrix;

__global__ void matMulKernel(Matrix A, Matrix B, Matrix C) {

	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	float Cval = 0;

	if (row < A.height && col < B.width) {
		for (int e=0; e<A.width; e++){
			Cval += A.elements[A.width * row + e] *
					B.elements[B.width * e + col];
		}
		C.elements[C.width * row + col] = Cval;
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


void print_matrix(Matrix A) {

	for(int i=0; i<A.height; i++){
		for(int j=0; j<A.width; j++){
			int idx = A.width * i + j;
			printf("Matrix[%d][%d] = %.4f\n", i, j, A.elements[idx]);
		}
	}
}

void check_results(Matrix A, Matrix B, Matrix C) {

	float value = 0;
	for(int i=0; i<A.height; i++){
		for(int j=0; j<B.width; j++){
			for(int k=0; k<B.height; k++){
				value += A.elements[A.width * i + k]
				       * B.elements[B.width * k + j];
			}
			printf("Host C[%d][%d] = %.4f\n", i, j, value);
			printf("diff %.4f\n", value - C.elements[C.width*i +j]);
			value = 0;
		}
	}

}
int main(int argc, char** argv) {
	
	Matrix h_A;
	h_A.height = 2;
	h_A.width = 5;
	size_t A_size = h_A.height * h_A.width * sizeof(float);
	h_A.elements = (float *)malloc(A_size);
	init_matrix(h_A.elements, h_A.width, h_A.height);

	Matrix h_B;
	h_B.height = 5;
	h_B.width = 2;
	size_t B_size = h_B.height * h_B.width * sizeof(float);
	h_B.elements = (float *)malloc(B_size);
	init_matrix(h_B.elements, h_B.width, h_B.height);

	Matrix h_C;
	h_C.height = h_A.height;
	h_C.width = h_B.width;
	size_t C_size = h_C.height * h_C.width * sizeof(float);
	h_C.elements = (float *)malloc(C_size);

	Matrix d_A;
	d_A.height = h_A.height;
	d_A.width = h_A.width;
	cudaMalloc(&d_A.elements, A_size);

	Matrix d_B;
	d_B.height = h_B.height;
	d_B.width = h_B.width;
	cudaMalloc(&d_B.elements, B_size);

	Matrix d_C;
	d_C.height = h_C.height;
	d_C.width = h_C.width;
	cudaMalloc(&d_C.elements, C_size);

	cudaMemcpy(d_A.elements, h_A.elements, A_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.elements, h_B.elements, B_size, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((h_B.width + BLOCK_SIZE -1)/BLOCK_SIZE, (h_A.height + BLOCK_SIZE -1)/BLOCK_SIZE);

	matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	cudaMemcpy(h_C.elements, d_C.elements, C_size, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	check_results(h_A, h_B, h_C);

	printf("Values of matrix A\n");
	print_matrix(h_A);
	printf("Values of matrix B\n");
	print_matrix(h_B);
	printf("Values of matrix C\n");
	print_matrix(h_C);

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);

	free(h_A.elements);
	free(h_B.elements);
	free(h_C.elements);

}

