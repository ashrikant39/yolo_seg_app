# Usage (CLI + C++ API)

> Build dependency note (Ubuntu): install TensorRT/OpenCV/Eigen packages first (see `README.md`), then configure/build with CMake.

## CLI

Folder-based (batched) inference:

```bash
./bin/yoloSegApp \
  --mode folder \
  --enginePath /path/to/model.engine \
  --videoDirPath /path/to/images \
  --saveDirPath /path/to/output_dir \
  --logPath /path/to/run.log \
  --logModelInfo true
```

Notes:

- Input images are discovered in `videoDirPath` by scanning for `.png` and `.jpg`.
- Output files are written to `saveDirPath`:
  - `<image_stem>_seg_vis.png`
  - `<image_stem>_det<i>_mask.png`

## C++ API

The main abstraction is `InferencePipeline`.

```cpp
#include "pipeline.hpp"

int main() {
  InferencePipeline pipeline(
      "/path/to/model.engine",
      "/path/to/run.log",
      "/path/to/images",
      "/path/to/output_dir",
      /*logModelInformation=*/true
  );

  pipeline.runInferencePipeline();
  return 0;
}
```

What the pipeline does:

1. Deserializes the `.engine`.
2. Allocates CUDA tensors (GPU/unified memory) for all I/O tensors.
3. Preprocesses images with `cv::dnn::blobFromImage` into an FP16 tensor.
4. Runs `enqueueV3()` using a CUDA stream.
5. Decodes outputs in `PostProcessor` and writes results to disk.

