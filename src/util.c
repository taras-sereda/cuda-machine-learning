#include <stdio.h>

void init(float*data, int n_elem) {
    for (int i = 0; i < n_elem; i++) {
        data[i] = (float) i;
    }
}

void print_tensor(float* data, int n_elem) {
    printf("\n");
	for(int i=0; i<n_elem; i++){
			printf("%.4f ", data[i]);
	}
    printf("\n");
}