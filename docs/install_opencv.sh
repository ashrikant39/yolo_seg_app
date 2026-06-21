#!/bin/bash

sudo apt update

sudo apt install -y build-essential cmake git pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libdc1394-22-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    libopenexr-dev \
    libatlas-base-dev gfortran \
    python3-dev python3-numpy

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

SRC_DIR="${1:-$HOME/opencv}"
BUILD_DIR="${2:-$SRC_DIR/build}"
INSTALL_DIR="${3:-$HOME/opencv_cuda}"

set -e

cmake -G Ninja \
    -S "$SRC_DIR" \
    -B "$BUILD_DIR" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D CUDA_NVCC_FLAGS="--std=c++17" \
    -D CMAKE_CUDA_FLAGS="--std=c++17" \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D WITH_CUDA=ON \
    -D WITH_NPP=OFF \
    -D CUDA_ARCH_BIN=7.5 \
    -D WITH_CUBLAS=ON \
    -D CUDA_FAST_MATH=ON \
    -D ENABLE_FAST_MATH=ON \
    -D WITH_TBB=ON \
    -D WITH_IPP=ON \
    -D WITH_OPENMP=ON \
    -D WITH_JPEG=ON \
    -D WITH_PNG=ON \
    -D WITH_TIFF=ON \
    -D WITH_WEBP=ON \
    -D BUILD_JPEG=OFF \
    -D BUILD_PNG=OFF \
    -D BUILD_ZLIB=OFF \
    -D BUILD_TIFF=OFF \
    -D BUILD_WEBP=OFF \
    -D BUILD_opencv_cudaarithm=ON \
    -D BUILD_opencv_cudaimgproc=ON \
    -D BUILD_opencv_cudawarping=ON \
    -D BUILD_opencv_cudafilters=ON \
    -D BUILD_opencv_cudev=ON \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON

cmake --build "$BUILD_DIR" --parallel "$(nproc)"
cmake --install "$BUILD_DIR"