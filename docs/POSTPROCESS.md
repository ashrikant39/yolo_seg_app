# Post-processing notes (YOLO-seg style)

This repository’s post-processor expects a YOLO-seg-like TensorRT engine with:

- a box output tensor named `boxes` (see `SimpleModelSettings` [`include/settings.hpp`](../include/settings.hpp))
- a mask output tensor named `masks`
- a objectness socre tensor named `objectness`
- a class label tensor named `classlabel`

See: [`modify_onnx.py`](../helpers/modify_onnx.py) to modify the onnx file returned from Ultralytics YOLO to generate a modified onnx file with the above outputs instead of the default output  tensors.

## Expected tensor interpretation

The current decoder assumes:

- `boxes`: rank-3, `[B, NObjects, 4]`
- `masks`: rank-4, `[B, NObjects, H, W]`
- `objectness`: rank-3 `[B, NObjects, 1]`
- `classlabel`: rank-3 `[B, NObjects, 1]`

## Tuning thresholds

The following values live in `include/utils/options.hpp`:

- `PostProcessingOptions::NMS_CONF_THRESH`
- `PostProcessingOptions::NMS_IOU_THRESH`
- `PostProcessingOptions::NMS_MAX_DET`

