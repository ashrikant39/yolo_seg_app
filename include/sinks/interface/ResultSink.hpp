#pragma once

#include <algorithm>

#include "logging/BaseLogger.hpp"
#include "core/cuda.hpp"
#include "post_process/utils/PostProcessUtils.hpp"

/**
 * @brief Abstract consumer for postprocessed detections.
 */
class ResultSink {
    public:
        virtual ~ResultSink() = default;

        /**
         * @brief Consume one frame's postprocess output.
         * @param output Mutable postprocess output for one frame.
         * @param logger Logger for diagnostics.
         */
        virtual void consumeSingle(PostProcessOutput& output, BaseLogger& logger) = 0;

        /**
         * @brief Consume a batch of postprocess outputs.
         * @param outputBatch Mutable outputs for a full batch.
         * @param logger Logger for diagnostics.
         */
        void consumeBatch(
            std::vector<PostProcessOutput>& outputBatch,
            BaseLogger& logger
        ) {
            
            for (auto& output : outputBatch ){
                consumeSingle(output, logger);
            }
        }
};
