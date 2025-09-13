#pragma once
#include "types/enums.hpp"


struct InferenceEngineOptions{
    static constexpr InferenceMode INFER_MODE = InferenceMode::ASYNC;
};


struct LoggerOptions{
    static constexpr const char* DEFAULT_LOG_FILE = "main.log";
};

struct PostProcessingOptions{
    static constexpr int NUM_CLASSES = 1;
    static constexpr float NMS_CONF_THRESH = 0.25f;
    static constexpr float NMS_IOU_THRESH = 0.45f;
    static constexpr int NMS_MAX_DET = 300;
};


struct VideoOptions{
    static constexpr double NORM_FACTOR_ADD_TO_SCALED = 0.0;
    static constexpr double NORM_FACTOR_SCALING_MUL = 1.0;
};
