__global__ void matMulKernel(float *A, float *B, float *C, int a_rows, int a_cols, int b_cols) {
	// Each thread computes one element of C.
	// by computing a dot product of the row of A and the column of B.
	// This method doesn't leverage data locality.


	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	float Cval = 0;
	if (row < a_rows && col < b_cols) {
		for (int i_idx=0; i_idx<a_cols; i_idx++) {
			Cval += A[a_cols * row + i_idx] * B[b_cols * i_idx + col];
		}
		C[b_cols * row + col] = Cval;
	}
}