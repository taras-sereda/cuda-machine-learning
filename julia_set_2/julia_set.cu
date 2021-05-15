// #include "../cuda_by_example/common/book.h"
#include "../cuda_by_example/common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex
{
    /* Struct for representing complex numbers */
    float real;
    float imag;

    __device__
    cuComplex( float a, float b): real(a), imag(b) {}

    __device__
    float magnitude2(void) {
        return real*real + imag*imag;
    }

    __device__
    cuComplex operator*(const cuComplex& a) {
        return cuComplex(real * a.real - imag * a.imag, imag * a.real + real * a.imag);
    }

    __device__
    cuComplex operator+(const cuComplex& a) {
        return cuComplex(real + a.real, imag + a.imag);
    }

};

__device__
int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

__global__
void kernel(unsigned char *ptr) {

    int x = blockIdx.x;
    int y = blockIdx.y;
    // multiply row idx by grid width (gridDim.x) and shift by col idx. gives a correct address.
    int offset = y * gridDim.x + x;
    int juliaVal = julia(x, y);

    ptr[offset*4 + 0] = 255 * juliaVal;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}



int main(void) {
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    cudaMalloc(
        (void**)&dev_bitmap,
        bitmap.image_size()
        );

    dim3 grid(DIM, DIM);
    kernel<<<grid,1>>>(dev_bitmap);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    bitmap.display_and_exit();
    cudaFree(dev_bitmap);

}