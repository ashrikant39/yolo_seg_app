#pragma once

#include "core/utils.hpp"

struct DefaultSettings {
    static constexpr char* DEFAULT_LOG_FILE = "main.log";
    static constexpr LoggingSeverity DEFAULT_LOG_SEVERITY = LoggingSeverity::INFO;
};

struct RawYoloSettings{
    static constexpr char* BOX_FEATURE_KEY = "output0";
    static constexpr char* PROTO_MASK_KEY = "output1";
};


struct SimplifiedYoloSettings{
    static constexpr char* BOX_KEY = "boxes";
    static constexpr char* MASK_KEY = "masks";
    static constexpr char* OBJECTNESS = "objectness";
    static constexpr char* CLASS_LABEL = "classlabel";
};


