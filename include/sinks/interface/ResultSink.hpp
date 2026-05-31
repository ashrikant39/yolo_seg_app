#pragma once

#include <algorithm>

#include "core/logger.hpp"
#include "core/cuda.hpp"
#include "post_process/utils/PostProcessUtils.hpp"

class ResultSink {
    public:
        virtual ~ResultSink() = default;
        virtual void consumeSingle(PostProcessOutput& output, Logger& logger) = 0;

        void consumeBatch(
            std::vector<PostProcessOutput>& outputBatch,
            Logger& logger
        ) {
            
            for (auto& output : outputBatch ){
                consumeSingle(output, logger);
            }
        }
};