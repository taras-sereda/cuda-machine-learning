import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.ops.gpu as plgpu
import numpy as np

# 1. Define the Pallas Kernel
def add_kernel(x_ref, y_ref, out_ref):
  pid = pl.program_id(0)
  jax.debug.print("program_id: {}", pid)
  # Pallas kernel operates on 'Refs' (memory references)
  # Block of size 128 (one per kernel instance)
  out_ref[...] = x_ref[...] + y_ref[...]

# 2. Define Kernel Wrapper Function
def pallas_add(x, y):
  return pl.pallas_call(
      add_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      grid=(x.shape[0] // 128,), # 1D Grid
      # block_spec defines how data is sliced for each grid index
      in_specs=[
          pl.BlockSpec((128,), index_map=lambda i: (i,)),
          pl.BlockSpec((128,), index_map=lambda i: (i,)),
          ],
      out_specs=pl.BlockSpec((128,), index_map=lambda i: (i,)),
  )(x, y)

# 3. Execute on GPU
# Create dummy data (must be divisible by block size, e.g., 128)
size = 128 * 10
x = jnp.ones(size, dtype=jnp.float32)
y = jnp.ones(size, dtype=jnp.float32) * 2

# JIT Compile and Run
result = jax.jit(pallas_add)(x, y)
print(result)

