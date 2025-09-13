#pragma once

#include <types/enums.hpp>

struct VideoSettings{
    static constexpr ChannelOrderMode CHANNEL_ORDER = ChannelOrderMode::BGR;
};

struct ModelSettings{
    static constexpr const char* BOX_FEATURE_KEY = "output0";
    static constexpr const char* PROTO_MASK_KEY = "output1";
};

struct ProcessingSettings{
    static constexpr ComputeDataType POSTPROCESSING_DTYPE = ComputeDataType::FLOAT32;
};
