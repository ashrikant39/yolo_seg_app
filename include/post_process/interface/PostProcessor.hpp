#pragma once

#include <vector>
#include <filesystem>
#include <opencv2/core.hpp>


#include "core/tensor.hpp"
#include "core/logger.hpp"
#include "core/options.hpp"
#include "core/cuda.hpp"
#include "core/metadata.hpp"
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
            const TensorViewMap& engineOutputBatch,
            std::vector<PostProcessOutput>& processedBatch,
            Logger& logger,
            cudaStream_t stream
        ) = 0;
};