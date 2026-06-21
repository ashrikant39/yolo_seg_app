#pragma once

#include <vector>
#include <filesystem>
#include <opencv2/core.hpp>


#include "core/tensor.hpp"
#include "core/cuda.hpp"
#include "logging/BaseLogger.hpp"
#include "post_process/utils/PostProcessUtils.hpp"


/**
 * @brief CPU-side Post Processor for YOLO-seg style TensorRT outputs.
 *
 * Responsibilities:
 * - decode boxes/scores/mask coefficients,
 * - run NMS,
 * - generate and save segmentation outputs.
 */
class PostProcessor {
    public:
        virtual ~PostProcessor() = default;

        virtual void process(
            const TensorViewMap& engineOutputViews,
            std::vector<PostProcessOutput>& processedBatch,
            BaseLogger& logger,
            cudaStream_t stream
        ) = 0;
};