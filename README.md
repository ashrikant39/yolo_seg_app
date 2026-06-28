# yolo_seg_app_cuda

TensorRT + CUDA + OpenCV application for running YOLO segmentation models from a
YAML configuration file. The executable is intentionally thin: `main.cpp` parses
the config path, constructs `Application`, and calls `Application::run()`.

## Current Run Model

The application is configured entirely from YAML:

```bash
./build/bin/yoloSegApp --config configs/YoloSegSimple.yaml
```

or positionally:

```bash
./build/bin/yoloSegApp configs/YoloSegSimple.yaml
```

The config controls:

- logging path and severity
- folder or video frame source
- preprocessing dimensions, dtype, scaling, and channel order
- TensorRT backend and serialized engine path
- memory groups for preprocessing, inference, and postprocessing
- input/output tensor names, shapes, dtypes, and IO modes
- postprocessing thresholds and output tensor offsets
- result sink mode and output directory

## Example Config

See `configs/YoloSegSimple.yaml`.

Important sections:

```yaml
frame_source:
  frameSourceType: folder        # folder or video
  frameSourcePath: assets/dummy_images_jpeg
  origImgHeight: 512
  origImgWidth: 1024
  batchSize: 1
```

```yaml
backend:
  inferenceBackendType: yolo_seg_trt
  modelType: yolo_segmentation
  outputType: yolo_modified_segmentation
  preferredInferenceDevice: gpu
  serializedModelPath: assets/engines/last_bs1_nms_modified_fp32.engine
```

```yaml
memory:
  preProcessingTensorGroups:
    - PinnedInput
  inferenceTensorGroups:
    - DeviceInput
    - DeviceOutput
  postProcessingTensorGroups:
    - PinnedOutput
    - HostPostProcessOutput
```

Tensor names in `inputTensorSpecs` and `outputTensorSpecs` must match the
TensorRT engine IO tensor names. For the modified YOLO segmentation path, the
CPU postprocessor currently expects these output names:

- `boxes`
- `masks`
- `classlabel`
- `objectness`

## Result Sinks

Save detections as binary:

```yaml
result_sink:
  resultsDir: assets/dummy_results_jpeg
  resultSinkType: save_detections
  saveDetMode: normalized
  drawDetMode: unset
  lineThickness: 1
```

Draw detections on images:

```yaml
result_sink:
  resultsDir: assets/dummy_results_latest_drawn
  resultSinkType: draw_detections
  saveDetMode: unset
  drawDetMode: contours_with_boxes
  lineThickness: 1
```

Supported drawing modes are configured by `DrawDetectionMode` in
`include/sinks/utils/enums.hpp`.

## Build

Prerequisites:

- C++17 compiler
- CMake 3.16+
- OpenCV development package
- CUDA Toolkit
- TensorRT
- `cxxopts`
- `yaml-cpp`
- `doctest` when `YOLO_BUILD_TESTS=ON`

Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If OpenCV is installed outside default CMake search paths:

```bash
OPENCV_INSTALL_PATH=/path/to/opencv \
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Build only CPU-safe tests and helper targets:

```bash
cmake -S . -B build/tests -DYOLO_BUILD_APP=OFF -DYOLO_BUILD_TESTS=ON
cmake --build build/tests -j
ctest --test-dir build/tests --output-on-failure
```

## Runtime Outputs

The application logs progress to `logging.logFilePath`. On completion, it logs:

- total source frames processed
- total batches processed
- elapsed wall-clock seconds
- total FPS

Output files are written under `result_sink.resultsDir`.

## C++ API

```cpp
#include "application/Application.hpp"

int main() {
    Application app("configs/YoloSegSimple.yaml");
    app.run();
    return 0;
}
```

## Additional Docs

- [`docs/USAGE.md`](docs/USAGE.md): short YAML-first usage notes
- [`docs/POSTPROCESS.md`](docs/POSTPROCESS.md): postprocessing assumptions
- [`docs/install.md`](docs/install.md): dependency installation notes
