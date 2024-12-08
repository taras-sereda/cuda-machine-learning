#include <stdio.h>
#include "io.h"
void print_file_info(char *filename, unsigned int size, unsigned int mode, unsigned int mtime) {
    printf("File: %s\n", filename);
    printf("Size: %d bytes\n", size);
    printf("Mode: %d\n", mode);
    printf("Modified time: %d\n\n", mtime);
}
