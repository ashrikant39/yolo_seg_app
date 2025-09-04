#pragma once

#include <types/enums.h>

struct VideoSettings{
    static constexpr ChannelOrderMode CHANNEL_ORDER = ChannelOrderMode::RGB;
};

struct ModelSettings{
    static constexpr const char* BOX_FEATURE_KEY = "output0";
    static constexpr const char* PROTO_MASK_KEY = "output1";
};

struct ProcessingSettings{
    static constexpr ComputeDataType POSTPROCESSING_DTYPE = ComputeDataType::FLOAT32;
};
