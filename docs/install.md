# Installation Guide

This document lists the system package installation steps required for building and running `yolo_seg_app`.

OpenCV is intentionally **not** installed using `apt install libopencv-dev` here, because this project uses a custom OpenCV build from source, typically with CUDA support. I installed OpenCV from source using [`install_opencv.sh`](install_opencv.sh) for my lightning studio environment.

---

## 1. Update apt package index

```bash
sudo apt-get update
```

---

## 2. Install basic build/runtime dependencies

```bash
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libcxxopts-dev
```

Optional utilities:

```bash
sudo apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates
```

---

## 3. Install TensorRT

Current tested TensorRT package version:

```text
10.16.1.11-1+cuda13.2
```

The expected installed packages include:

```text
libnvinfer10
libnvinfer-dev
libnvinfer-plugin10
libnvinfer-plugin-dev
libnvonnxparsers-dev
libnvinfer-bin
```

### Option A: Install from an NVIDIA local repository `.deb`

Place the TensorRT local repository package under:

```text
assets/
```

The link to the downloadable package:

```text
wget https://developer.download.nvidia.com/compute/tensorrt/10.16.1/local_installers/nv-tensorrt-local-repo-ubuntu2204-10.16.1-cuda-13.2_1.0-1_amd64.deb
```

Then run:

```bash
sudo dpkg -i assets/nv-tensorrt-local-repo-ubuntu2204-10.16.1-cuda-13.2_1.0-1_amd64.deb
```

Copy the repository keyring. The exact key filename may differ, so inspect the directory first:

```bash
ls /var/nv-tensorrt-local-repo-ubuntu2204-10.16.1-cuda-13.2/
```

Then copy the `.gpg` keyring file:

```bash
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.16.1-cuda-13.2/*.gpg /usr/share/keyrings/
```

Update apt:

```bash
sudo apt-get update
```

Install TensorRT packages:

```bash
sudo apt-get install -y \
    tensorrt \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    libnvonnxparsers-dev \
    libnvinfer-bin
```

---

## 4. Verify TensorRT installation

Check installed TensorRT packages:

```bash
dpkg -l | grep nvinfer
```

Expected version pattern:

```text
10.16.1.11-1+cuda13.2
```

Check `trtexec`:

```bash
trtexec --version
```

Check CUDA compiler version:

```bash
nvcc -V
```

or:

```bash
nvcc -v
```

Optional Python TensorRT check, if the Python package is installed:

```bash
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

---

## 5. OpenCV

This project expects a custom OpenCV build, usually installed from source with CUDA support.

Set the OpenCV install path using an environment variable, for example:

```bash
export OPENCV_CUDA_INSTALL_PATH=$HOME/opencv_cuda
```

Then configure the project CMake build to use this path.

Example:

```bash
cmake -S . -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D OpenCV_DIR=$OPENCV_CUDA_INSTALL_PATH/lib/cmake/opencv4
```

---

## 6. Build the project

```bash
cmake -S . -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D OpenCV_DIR=$OPENCV_CUDA_INSTALL_PATH/lib/cmake/opencv4

cmake --build build -j$(nproc)
```

---

## 7. Notes

TensorRT `.engine` files are not fully portable. They are tied to the TensorRT version, CUDA version, GPU architecture, and sometimes driver/runtime details.

If a prebuilt `.engine` fails to load, rebuild the engine on the target machine from the ONNX model using the installed TensorRT version.
