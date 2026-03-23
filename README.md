# yolo_seg_app

TensorRT + CUDA + OpenCV application for running YOLO-seg style models and decoding:

- detection boxes (NMS)
- instance segmentation masks (prototype + mask coefficients)

The project is currently in progress; the CLI supports **folder (batched) inference**.

## What this repo uses

- **NVIDIA TensorRT**: loads a serialized `.engine`, creates an execution context, binds tensor addresses, and runs inference.
- **CUDA**: manages the CUDA stream used by TensorRT and handles FP16 unified memory prefetching.
- **OpenCV**: image loading (`imread`), preprocessing (`blobFromImage`), NMS (`cv::dnn::NMSBoxes`), and visualization/output writing.
- **Eigen**: CPU-side tensor view helpers (used for decoding via `slice`/`chip`).
- **cxxopts**: command-line argument parsing.

## Outputs

For each input image, the post-processor writes (to `--saveDirPath` / `saveDirPath`):

- `<image_stem>_seg_vis.png`: visualization overlay of predicted masks
- `<image_stem>_det<i>_mask.png`: raw instance masks for each selected detection `i` (after NMS)

## Environment variables (required)

See: [`docs/ENV_VARS.md`](docs/ENV_VARS.md)

## Post-processing assumptions

See: [`docs/POSTPROCESS.md`](docs/POSTPROCESS.md)

## Installation

### Prerequisites

You need:

- C++17 toolchain
- CMake
- OpenCV (built/installed such that CMake can find it)
- Eigen (CMake config available via `Eigen_CMAKE_DIR`)
- TensorRT installed (and accessible via `TRT_ROOT`)
- CUDA toolkit installed (CMake needs `CUDA_TOOLKIT_ROOT_DIR`)
- `cxxopts` available to CMake (the project expects a CMake package target `cxxopts::cxxopts`)

### Build (CMake)

Example out-of-tree build:

```bash
mkdir -p build
cd build

cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTRT_ROOT=/path/to/TensorRT \
  -DEigen_CMAKE_DIR=/path/to/eigen/cmake/config \
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda \
  ..

cmake --build . -j
```

## Running (CLI)

The current binary (`yoloSegApp`) expects a TensorRT engine and an input directory of images.

### Folder (batched) inference

```bash
./bin/yoloSegApp \
  --mode folder \
  --enginePath /path/to/model.engine \
  --videoDirPath /path/to/images \
  --saveDirPath /path/to/output_dir \
  --logPath /path/to/run.log \
  --logModelInfo true
```

Supported image extensions: `.png`, `.jpg` (folder scanning is done via directory iteration).

### Video inference

`--mode video` is accepted by the CLI but currently prints that it is not implemented yet.

See next steps in [`docs/USAGE.md`](docs/USAGE.md) if you want to extend the input source to `batchSize=1`.

## Using the C++ API

See: [`docs/USAGE.md`](docs/USAGE.md)

