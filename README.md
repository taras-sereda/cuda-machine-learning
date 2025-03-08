# GPU Programming

## Build

1. To build cuda examples, run: `make`. Compiled binaries will be saved in `bin` directory. To clean up, run `make clean`.
2. For triton kernels python vitual env should be created.

```bash
uv venv --python=python3.13
source .venv/bin/activate
pip install -r requirements.txt
```

To recompile python requiremenets: `uv pip compile requirements.in -o requirements.txt`

## Run samples

1. CUDA
   - Device info: `./bin/device_info`
   - Dynamic parallelism: `./bin/dynamic_parallelism`
2. Triton
   - Vector addition: `python src/triton_kernels/vector_add.py`

## General purpose GPU programming and related resources

1. [ZLUDA](https://github.com/vosen/ZLUDA) CUDA-like programming model for AMD GPUs, written in Rust.
2. [Hemi](https://github.com/harrism/hemi) (No longer maintened) library for writting reusable CPU and GPU code. Single kernel function executable on both device types. More in this [blog post](https://developer.nvidia.com/blog/simple-portable-parallel-c-hemi-2/).
3. [CUDA device management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html) functions refference guide.
4. [Holistic Trace Analysis](https://hta.readthedocs.io/en/latest/#) performance analysis of CUDA kernels. Allows to collect metrics on kernel execution time for distributed training systems.

## References

1. FlashInfer - kernel library with focus on LLM-inference [code](https://github.com/flashinfer-ai/flashinfer) [paper](https://arxiv.org/abs/2501.01005)
