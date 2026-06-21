#pragma once

#include <vector>
#include <filesystem>

#include "post_process/interface/PostProcessor.hpp"
#include "post_process/config/PostProcessorConfig.hpp"

namespace fs = std::filesystem;

/**
 * @brief CPU-side Simple post-processing for YOLO-seg style TensorRT outputs.
 *
 * Responsibilities:
 * - convert output tensor buffers from FP16 to FP32 host buffers,
 * - decode boxes/scores/mask coefficients,
 * - run NMS,
 * - generate and save segmentation outputs.
 */
class YoloSegCpuPostProcessorRaw : public PostProcessor {

    public:
        explicit YoloSegCpuPostProcessorRaw(const PostProcessorConfig& config);

        void process(
            const TensorViewMap& engineOutputViews,
            std::vector<PostProcessOutput>& processedBatch,
            BaseLogger& logger,
            cudaStream_t stream
        ) override;
};
