#pragma once

#include <vector>
#include <filesystem>

#include "post_process/interface/PostProcessor.hpp"
#include "post_process/config/PostProcessorConfig.hpp"

namespace fs = std::filesystem;

/**
 * @brief GPU-side postprocessor for modified YOLO segmentation outputs.
 */
class YoloSegGpuPostProcessorSimple : public PostProcessor {

    public:
        /**
         * @brief Creates a GPU modified-output segmentation postprocessor from YAML-derived settings.
         */
        explicit YoloSegGpuPostProcessorSimple(const PostProcessorConfig& config);

        /**
         * @brief Converts model output tensors into per-frame detections.
         */
        void process(
            const TensorViewMap& engineOutputViews,
            std::vector<PostProcessOutput>& processedBatch,
            BaseLogger& logger,
            cudaStream_t stream
        ) override;
};
