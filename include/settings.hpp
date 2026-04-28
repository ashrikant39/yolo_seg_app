#pragma once

#include <utils/enums.hpp>
#include <utils/options.hpp>

struct VideoSettings{
    static constexpr ChannelOrderMode CHANNEL_ORDER = ChannelOrderMode::BGR;
};

struct ModelSettings{
    static constexpr const char* BOX_FEATURE_KEY = "output0";
    static constexpr const char* PROTO_MASK_KEY = "output1";
};


struct SimpleModelSettings{
    static constexpr const char* BOX_KEY = "boxes";
    static constexpr const char* MASK_KEY = "masks";
    static constexpr const char* OBJECTNESS = "objectness";
    static constexpr const char* CLASS_LABEL = "classlabel";
};


struct ProcessingSettings{
    static constexpr ComputeDataType POSTPROCESSING_DTYPE = ComputeDataType::FLOAT32;
};

/** YOLO-seg head layout: [cx,cy,w,h] + optional obj + class scores + nm mask coeffs (Ultralytics-style). */
struct YoloSegDecodeSettings{
    /** If true, first 4 channels are center-x, center-y, w, h in model input pixels. If false, xyxy in pixels. */
    static constexpr bool BOX_IS_XYWH = true;
    /** Index where mask coefficients start (after box + obj + per-class scores). */
    static constexpr int MASK_COEFF_START = 4 + 1 + PostProcessingOptions::NUM_CLS_DIMS;
};
