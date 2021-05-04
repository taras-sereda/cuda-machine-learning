1. Compile CUDA kernels
```
nvcc device_info.cu -o device_info
```

2. Theory
Thread Hierarchy with easy examples.
```
1d(Dx=15)
like finiding a point on a line
threadIdx = (7)
threadId = threadIdx.x
threadId = 7

-> -------x-------

2d(Dx=15, Dy=4)
like finding a line on a plane and then finding a point on it.
threadIdx = (2, 7)
threadId = threadIdx.x + threadIdx.y * Dx
threadId = 7 + 2 * 15 = 37
   ---------------
   ---------------
-> -------x-------
   ---------------



3d(Dx=15, Dy=4, Dz=2)
threadIdx = (0, 2, 7)
threadId = threadIdx.x + threadIdx.y * Dx + threadIdx.z * Dx * Dy
like finding a plane in a volume then line on a plane and finaly a point.

   --------------- --
   ---------------   | first
-> --------x------   | find a slice
   --------------- --

   ---------------
   --------------- 
   --------------- 
   --------------- 
```
