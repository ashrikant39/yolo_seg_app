# yolo_seg_app

TensorRT + CUDA + OpenCV application for running YOLO-seg style models and decoding:

- detection boxes (NMS)
- instance segmentation masks (prototype + mask coefficients)

The project is currently in progress; the CLI supports **folder (batched) inference**.

The model files ("last_bs1.engine" and "last_bs1_nms_modified_fp32.engine") can be downloaded from [Asset Folder.](https://drive.google.com/drive/folders/1om3BMPzTnvqPSztqVBbbErkvHvNLsRlj?usp=drive_link)

## What this repo uses

- **NVIDIA TensorRT**: loads a serialized `.engine`, creates an execution context, binds tensor addresses, and runs inference.
- **CUDA**: manages the CUDA stream used by TensorRT and handles FP16 unified memory prefetching.
- **OpenCV**: image loading (`imread`), preprocessing (`blobFromImage`), NMS (`cv::dnn::NMSBoxes`), and visualization/output writing.
- **cxxopts**: command-line argument parsing.

## Outputs

For each input image, the post-processor writes (to `--saveDirPath` / `saveDirPath`):

- `<image_stem>_seg_vis.png`: visualization overlay of predicted masks
- `<image_stem>_det<i>_mask.png`: raw instance masks for each selected detection `i` (after NMS)

## Environment variables

No environment variables are required for a standard Ubuntu apt-based setup.
See: [`docs/ENV_VARS.md`](docs/ENV_VARS.md)

## Post-processing assumptions

See: [`docs/POSTPROCESS.md`](docs/POSTPROCESS.md)

## Installation

### Prerequisites

You need:

- C++17 toolchain
- CMake
- OpenCV development package
- TensorRT 10.x + CUDA toolkit
- `cxxopts` available to CMake (the project expects a CMake package target `cxxopts::cxxopts`)

### Ubuntu 22.04 installation commands

The following commands match your TensorRT 10.0.0 + CUDA 12.4 setup:

```bash
sudo apt-get update

sudo apt-get install -y libopencv-dev

sudo dpkg -i assets/nv-tensorrt-local-repo-ubuntu2204-10.0.0-cuda-12.4_1.0-1_amd64.deb

sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.0.0-cuda-12.4/nv-tensorrt-local-2B368663-keyring.gpg /usr/share/keyrings/

sudo apt-get update

sudo apt-get install -y tensorrt libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev libnvinfer-bin

sudo apt-get install -y libcxxopts-dev
```

### Build (CMake)

Example out-of-tree build:

```bash
mkdir -p build
cd build

cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  ..

cmake --build . -j
```

The default CMake configuration assumes standard Ubuntu library/include locations (`/usr/include`, `/usr/lib/*`) and does not require extra environment variables.

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

