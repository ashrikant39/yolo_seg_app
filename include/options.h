/*
All options should be of a basic type
*/ 
#pragma once
#include "enums.h"


struct VideoOptions{
    static constexpr int BATCH_SIZE = 16;
    static constexpr int IMAGE_HEIGHT = 2048;
    static constexpr int IMAGE_WIDTH = 2048;
    static constexpr double NORM_FACTOR_ADD_TO_SCALED = 0.0f;
    static constexpr double NORM_FACTOR_SCALING_MUL = 1.0f/255.0f;
    static constexpr ComputeDataType OUTPUT_DTYPE = ComputeDataType::FLOAT16;
};


struct InferenceEngineOptions{
    static constexpr InferenceMode INFER_MODE = InferenceMode::ASYNC;
    static constexpr ComputeDataType INFER_DTYPE = ComputeDataType::FLOAT16;
};

struct PostProcessingOptions{
    static constexpr ComputeDataType POST_PROCESS_DTYPE = ComputeDataType::FLOAT32;
    static constexpr int NUM_CLASSES = 1;
    static constexpr float NMS_CONF_THRESH = 0.25f;
    static constexpr float NMS_IOU_THRESH = 0.45f;
    static constexpr int NMS_MAX_DET = 300;
};