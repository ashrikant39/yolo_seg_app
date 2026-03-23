# Environment Variables

This project’s CMake expects a few environment variables / CMake cache entries to locate CUDA, TensorRT, and Eigen.

## Required

### `TRT_ROOT`

- Used to locate TensorRT headers (`${TRT_ROOT}/include`) and libraries.
- How CMake gets it:
  - if `-DTRT_ROOT=...` is provided, that wins
  - otherwise it uses `ENV{TRT_ROOT}`

Example:

```bash
export TRT_ROOT=/path/to/TensorRT
```

or at configure time:

```bash
cmake -DTRT_ROOT=/path/to/TensorRT ..
```

### `CUDA_TOOLKIT_ROOT_DIR`

- `find_package(CUDA REQUIRED)` (via CMake’s legacy FindCUDA module) expects this.

Example:

```bash
cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..
```

### `Eigen_CMAKE_DIR`

- `Eigen3_DIR` is read from `ENV{Eigen_CMAKE_DIR}`.

Example:

```bash
export Eigen_CMAKE_DIR=/path/to/eigen
```

## Optional / Helpful

### NVTX / profiling

The code uses NVTX ranges when compiled without `NDEBUG`. If you are profiling with Nsight Systems, make sure your environment supports NVTX.

