#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];

    // Step 1: Allocate and initialize a JPEG decompression object
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    // Step 2: Specify the source of the compressed data (e.g., a file)
    FILE *infile = fopen(filename, "rb");
    if (infile == NULL) {
        fprintf(stderr, "Cannot open %s\n", filename);
        return 1;
    }

    jpeg_stdio_src(&cinfo, infile);

    // Step 3: Read file parameters with jpeg_read_header()
    jpeg_read_header(&cinfo, TRUE);

    // Step 4: Set parameters for decompression
    // (We don't need to change any of the defaults set by jpeg_read_header)

    // Step 5: Start decompressor
    jpeg_start_decompress(&cinfo);

    // Step 6: Process data
    int row_stride = cinfo.output_width * cinfo.output_components;
    printf("width: %d, height: %d, componets: %d\n", cinfo.output_width, cinfo.output_height, cinfo.output_components);
    unsigned char *buffer = (unsigned char *)malloc(row_stride * cinfo.output_height);

    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *row_pointer = buffer + cinfo.output_scanline * row_stride;
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
    }

    // Step 7: Finish decompression
    jpeg_finish_decompress(&cinfo);

    // Step 8: Release the JPEG decompression object
    jpeg_destroy_decompress(&cinfo);

    fclose(infile);

    // Now you have the image data in the buffer
    // You can use it as needed in your application

    // Example: Print the first 10 bytes
    for (int i = 0; i < 10; i++) {
        printf("%02x ", buffer[i]);
    }
    printf("\n");

    // Free the buffer when done
    free(buffer);

    return 0;
}