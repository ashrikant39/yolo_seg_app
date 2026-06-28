#pragma once

#include <vector>
#include <filesystem>

#include "post_process/interface/PostProcessor.hpp"
#include "post_process/config/PostProcessorConfig.hpp"

namespace fs = std::filesystem;

/**
 * @brief CPU-side postprocessor for YOLO detection outputs.
 */
class YoloDetCpuPostProcessor : public PostProcessor {

    public:
        /**
         * @brief Creates a detection postprocessor from YAML-derived settings.
         */
        explicit YoloDetCpuPostProcessor(const PostProcessorConfig& config);

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
