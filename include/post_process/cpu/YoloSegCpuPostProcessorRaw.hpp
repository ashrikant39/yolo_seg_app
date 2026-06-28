#pragma once

#include <vector>
#include <filesystem>

#include "post_process/interface/PostProcessor.hpp"
#include "post_process/config/PostProcessorConfig.hpp"

namespace fs = std::filesystem;

/**
 * @brief CPU-side postprocessor for raw YOLO segmentation outputs.
 */
class YoloSegCpuPostProcessorRaw : public PostProcessor {

    public:
        /**
         * @brief Creates a raw segmentation postprocessor from YAML-derived settings.
         */
        explicit YoloSegCpuPostProcessorRaw(const PostProcessorConfig& config);

        /**
         * @brief Converts raw model output tensors into per-frame detections.
         */
        void process(
            const TensorViewMap& engineOutputViews,
            std::vector<PostProcessOutput>& processedBatch,
            BaseLogger& logger,
            cudaStream_t stream
        ) override;
};
