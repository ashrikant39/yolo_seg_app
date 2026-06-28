#pragma once

#include <vector>
#include <filesystem>
#include <opencv2/core.hpp>


#include "core/tensor.hpp"
#include "core/cuda.hpp"
#include "logging/BaseLogger.hpp"
#include "post_process/utils/PostProcessUtils.hpp"


/**
 * @brief Abstract interface for converting inference outputs into detections.
 */
class PostProcessor {
    public:
        virtual ~PostProcessor() = default;

        /**
         * @brief Postprocess model output tensors.
         * @param engineOutputViews Output tensor views keyed by model tensor name.
         * @param processedBatch Batch outputs containing frame metadata and detections.
         * @param logger Logger for diagnostics.
         * @param stream CUDA stream used by GPU implementations, or nullptr for CPU paths.
         */
        virtual void process(
            const TensorViewMap& engineOutputViews,
            std::vector<PostProcessOutput>& processedBatch,
            BaseLogger& logger,
            cudaStream_t stream
        ) = 0;
};
