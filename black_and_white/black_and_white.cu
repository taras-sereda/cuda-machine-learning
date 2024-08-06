// build
// nvcc black_and_white.cu -o black_and_white -ljpeg

// execute
// black_and_white <img_path.jpg>

#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <unistd.h>

int const N_CHANNELS = 3;

__global__ void rgb_to_grayscale_kernel(unsigned char *im_out, unsigned char *im_in, int width, int height, int n_channel)
{
    int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int row_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if (col_idx < width && row_idx < height)
    {
        int gray_offset = width * row_idx + col_idx;
        int rgb_offset = gray_offset * n_channel;

        unsigned char r = im_in[rgb_offset];
        unsigned char g = im_in[rgb_offset + 1];
        unsigned char b = im_in[rgb_offset + 2];

        im_out[gray_offset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

void save_image(const char *filename, unsigned char *image_buffer, int width, int height, J_COLOR_SPACE color_space, int n_channel)
{
    // More examples
    // lib's github: https://github.com/LuaDist/libjpeg/blob/master/example.c
    // blog: https://www.tspi.at/2020/03/20/libjpegexample.html
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE *outfile;
    JSAMPROW row_pointer[1];

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(filename, "wb")) == NULL)
    {
        fprintf(stderr, "Can't open file %s\n", filename);
        jpeg_destroy_compress(&cinfo);
        exit(1);
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = n_channel;
    cinfo.in_color_space = color_space;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 100, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    // Inefficient, requires fliping and backward reading. Though still ok for my needs.
    unsigned char *flipped_row = (unsigned char *)malloc(width * sizeof(unsigned char));
    for (int y = height - 1; y >= 0; y--)
    {
        // Reverse the order of pixels in each row
        for (int x = 0; x < width; x++)
        {
            flipped_row[x] = image_buffer[y * width + (width - 1 - x)];
        }
        row_pointer[0] = flipped_row;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
    // should be way faster, byt saved images are rotated.
    // int row_stride = width * n_channel;
    // while (cinfo.next_scanline < cinfo.image_height) {
    //     /* jpeg_write_scanlines expects an array of pointers to scanlines.
    //     * Here the array is only one element long, but you could pass
    //     * more than one scanline at a time if that's more convenient.
    //     */
    //     row_pointer[0] = & image_buffer[cinfo.next_scanline * row_stride];
    //     (void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
    // }
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    FILE *infile = fopen(filename, "rb");
    if (infile == NULL)
    {
        fprintf(stderr, "Cannot open %s\n", filename);
        return 1;
    }

    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    int row_stride = cinfo.output_width * cinfo.output_components;
    // printf("width: %d, height: %d, componets: %d\n", cinfo.output_width, cinfo.output_height, cinfo.output_components);
    unsigned char *rgb_buffer = (unsigned char *)malloc(row_stride * cinfo.output_height);

    while (cinfo.output_scanline < cinfo.output_height)
    {
        unsigned char *row_pointer = rgb_buffer + cinfo.output_scanline * row_stride;
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    fclose(infile);

    // Example: Print the first 10 bytes
    // for (int i = 0; i < 10; i++)
    // {
    //     printf("%02x ", rgb_buffer[i]);
    // }
    // printf("\n");

    int const width = cinfo.output_width;
    int const height = cinfo.output_height;
    size_t gray_size = width * height * sizeof(unsigned char);
    size_t rgb_size = gray_size * N_CHANNELS;
    unsigned char *gray_buffer = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (gray_buffer == NULL)
    {
        fprintf(stderr, "Memory allocation failed");
        return 1;
    }
    // for (int i = 0; i < width * height; i++)
    // {
    //     imageBuffer[i] = 128;
    // }

    unsigned char *im_in_d, *im_out_d;

    // Allocate memory on device
    cudaError_t alloc_err_im_in = cudaMalloc(&im_in_d, rgb_size);
    cudaError_t alloc_err_im_out = cudaMalloc(&im_out_d, gray_size);

    // Copy data from host to device
    cudaMemcpy(im_in_d, rgb_buffer, rgb_size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
    dim3 dimBlock(16, 16, 1);
    rgb_to_grayscale_kernel<<<dimGrid, dimBlock>>>(im_out_d, im_in_d, width, height, N_CHANNELS);

    // Copy data from device to host
    cudaMemcpy(gray_buffer, im_out_d, gray_size, cudaMemcpyDeviceToHost);

    // Free device buffers
    cudaFree(im_in_d);
    cudaFree(im_out_d);

    // save_image("out.jpg", rgb_buffer, width, height, JCS_RGB, 3);
    save_image("out.jpg", gray_buffer, width, height, JCS_GRAYSCALE, 1);

    // // Example: Print the first 10 bytes
    // for (int i = 0; i < 10; i++)
    // {
    //     printf("%d ", rgb_buffer[i]);
    // }
    // printf("\n");

    // // Example: Print the first 10 bytes
    // for (int i = 0; i < 10; i++)
    // {
    //     printf("%d ", gray_buffer[i]);
    // }
    // printf("\n");

    // Free host buffers
    free(gray_buffer);
    free(rgb_buffer);

    // sleep(10);

    return 0;
}