### Notes

1. Build. Manual build is a 2 stage process now. First a C part is built and `io.o` file saved. Next cuda c++ part is built and `io.o` object is explicitly linked, togather with "tar" lib.

```
gcc -c io.c -Wall -fPIC -o io.o
nvcc -o data_transfer data_transfer.cu io.o -Xcompiler "-Wall" -ltar
```

2. Usage:
```
./data_transfer <tar_file>
```