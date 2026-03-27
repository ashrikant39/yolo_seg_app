# Environment Variables

For this repository’s Ubuntu setup, **no environment variables are required**.

If you install dependencies from apt as documented in `README.md`, CMake discovers:

- OpenCV from standard system include/lib locations
- TensorRT libraries (`nvinfer`, `nvinfer_plugin`, `nvonnxparser`) from `/usr/lib/*`
- CUDA toolkit via `find_package(CUDAToolkit)`

## NVTX / profiling note

The code uses NVTX ranges in non-release builds. For Nsight profiling, ensure your runtime/toolkit setup provides NVTX support.

