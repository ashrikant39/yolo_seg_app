# Post-processing notes (YOLO-seg style)

This repository’s post-processor expects a YOLO-seg-like TensorRT engine with:

- a box/detection output tensor named `output0` (see `include/settings.hpp`)
- a prototype mask output tensor named `output1` (see `include/settings.hpp`)

See: `include/settings.hpp`

## Expected tensor interpretation

The current decoder assumes:

- `output0`: rank-4, `[B, N, C, 1]`
  - first 4 values per row are box parameters (`cx, cy, w, h` in model input pixels by default)
  - remaining values contain objectness/classes (depending on `MASK_COEFF_START`) and then mask coefficients
- `output1`: rank-4, `[B, nm, H, W]`
  - these are prototype masks

Instance mask is computed from:

- coefficients from the detection row
- prototypes from `output1`

Then the mask is:

1. sigmoid + threshold
2. resized to the model input resolution
3. saved and/or overlaid in the visualization image

## Tuning thresholds

The following values live in `include/utils/options.hpp`:

- `PostProcessingOptions::NMS_CONF_THRESH`
- `PostProcessingOptions::NMS_IOU_THRESH`
- `PostProcessingOptions::NMS_MAX_DET`

