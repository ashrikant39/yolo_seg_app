#pragma once

#include <algorithm>

#include "logging/BaseLogger.hpp"
#include "core/cuda.hpp"
#include "post_process/utils/PostProcessUtils.hpp"

class ResultSink {
    public:
        virtual ~ResultSink() = default;
        virtual void consumeSingle(PostProcessOutput& output, BaseLogger& logger) = 0;

        void consumeBatch(
            std::vector<PostProcessOutput>& outputBatch,
            BaseLogger& logger
        ) {
            
            for (auto& output : outputBatch ){
                consumeSingle(output, logger);
            }
        }
};